import dataclasses
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from src.hippox.main import Hippo
from functools import partial
from tqdm import tqdm
from typing import Tuple, NamedTuple, MutableMapping, Any


def discretize(A, B, delta_t, mode="zoh"):
    if mode == "bilinear":
        num, denom = 1 + 0.5 * delta_t*A, 1 - 0.5 * delta_t*A
        return num / denom, delta_t * B / denom
    elif mode == "zoh":
        return jnp.exp(delta_t*A), (jnp.exp(delta_t*A)-1) / A*B


def s4d_ssm(A, B, C, delta_t):
    N = A.shape[0]
    _A, _B = discretize(A, B, delta_t, mode="zoh")
    _A = jnp.diag(_A)
    _B = _B.reshape(N, 1)
    _C = C.reshape(1, N)
    return _A, _B, _C


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


@partial(jax.jit, static_argnums=3)
def s4d_kernel_zoh(A, C, delta_t, L):
    kernel_l = lambda l: (C * (jnp.exp(delta_t * A) - 1) / A * jnp.exp(l * delta_t * A)).sum()
    return jax.vmap(kernel_l)(jnp.arange(L)).real


def causal_convolution(u, K):
    assert K.shape[0] == u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(shape, dtype):
        uniform = hk.initializers.RandomUniform()
        return uniform(shape, dtype)*(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
    return init


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


class S4D(hk.Module):
    def __init__(self,
                 state_size: int,
                 seq_length: int,
                 inference_mode: bool = False
    ):
        super(S4D, self).__init__()
        self._state_size = state_size
        self._inference_mode = inference_mode

        _hippo = Hippo(self._state_size, 'legs')
        _hippo_params = _hippo()
        self._lambda_real = hk.get_parameter(
            'lambda_real',
            [self._state_size,],
            init=_hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            [self._state_size,],
            init=_hippo.lambda_initializer('imaginary')
        )
        self._lambda = jnp.clip(self._lambda_real, None, -1e-4) + 1j * self._lambda_imag

        self._c = hk.get_parameter(
            'c',
            [self._state_size, 2],
            init=hk.initializers.RandomNormal(stddev=0.5**0.5)
        )
        self._c = self._c[..., 0] + 1j * self._c[..., 1]

        self._d = hk.get_parameter(
            'd',
            [1,],
            init=jnp.ones,
        )

        self._delta = hk.get_parameter(
            'delta',
            [1,],
            dtype=jnp.float32,
            init=log_step_initializer()
        )
        self._timescale = jnp.exp(self._delta)

        if not self._inference_mode:
            self._kernel = s4d_kernel_zoh(self._lambda, self._c, self._timescale, seq_length)
        else:
            self._ssm = s4d_ssm(self._lambda, jnp.ones(self._state_size), self._c, self._timescale)
            self._state = hk.get_state('state', [self._state_size], jnp.complex64, jnp.zeros)

    def __call__(self, u):
        if not self._inference_mode:
            return causal_convolution(u, self._kernel) + self._d * u
        else:
            x_k, y_s = scan_SSM(*self._ssm, u[:, jnp.newaxis], self._state)
            hk.set_state('state', x_k)
            return y_s.reshape(-1).real + self._d * u


@dataclasses.dataclass
class S4Block(hk.Module):
    s4_layer: S4D
    d_model: int
    dropout_rate: float
    prenorm: bool = True
    glu: bool = True
    istraining: bool = False
    inference_mode: bool = False

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = layer_norm(x)
        x = hk.vmap(self.s4_layer, in_axes=1, out_axes=1, split_rng=True)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if self.glu:
            x = hk.Linear(self.d_model)(x) * jax.nn.sigmoid(hk.Linear(self.d_model)(x))
        else:
            x = hk.Linear(self.d_model)(x)
        x = skip + hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if not self.prenorm:
            x = layer_norm(x)
        return x


@dataclasses.dataclass
class Embedding(hk.Module):
    n_embeddings: int
    n_features: int

    def __call__(self, x):
        y = hk.Embed(self.n_embeddings, self.n_features)(x[..., 0])
        return jnp.where(x > 0, y, 0.0)


@dataclasses.dataclass
class StackedS4(hk.Module):
    s4_layer: S4D
    d_model: int
    n_layers: int
    d_output: int
    prenorm: bool = True
    dropout_rate: float = 0.0
    embedding: bool = False
    classification: bool = False
    istraining: bool = True
    inference_mode: bool = False

    def __post_init__(self):
        super(StackedS4, self).__post_init__()
        if self.embedding:
            self._encoder = Embedding(self.d_output, self.d_model)
        else:
            self._encoder = hk.Linear(self.d_model)
        self._decoder = hk.Linear(self.d_output)
        self._layers = [
            S4Block(
                s4_layer=self.s4_layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                istraining=self.istraining,
                inference_mode=self.inference_mode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0
            if not self.inference_mode:
                x = jnp.pad(x[:-1], [(1, 0), (0, 0)])
        x = self._encoder(x)
        for layer in self._layers:
            x = layer(x)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self._decoder(x)
        return jax.nn.log_softmax(x, axis=-1)


_Metrics = MutableMapping[str, Any]

STATE_SIZE: int = 64
D_MODEL: int = 128
N_LAYERS: int = 4
EPOCHS: int = 10
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-1
WEIGHT_DECAY: float = 1e-2


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
def create_mnist_dataset(batch_size=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train_set = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM



# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(batch_size=128):
    print("[*] Generating MNIST Classification Dataset...")
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train_set = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test_set = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "mnist": create_mnist_dataset,
    "mnist-classification": create_mnist_classification_dataset,
}


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jnp.ndarray
    step: jnp.ndarray


@partial(jax.jit, static_argnums=0)
def init_state(
        model: hk.transform,
        rng: jnp.ndarray,
        init_data: jnp.ndarray
) -> TrainingState:

    rng, init_rng = jax.random.split(rng)
    initial_params = model.init(init_rng, init_data)
    initial_opt_state = optimizer.init(initial_params)

    return TrainingState(
        params=initial_params,
        opt_state = initial_opt_state,
        rng_key=rng,
        step=jnp.array(0)
    )


optimizer = optax.adam(LEARNING_RATE)
# optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_epoch(
        state: TrainingState,
        trainloader: DataLoader,
        model: hk.transform,
        classification: bool = False,
) -> Tuple[TrainingState, jnp.ndarray, jnp.ndarray]:

    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        state, metrics = train_step(
            state, inputs, targets,
            model, classification
        )
        batch_losses.append(metrics['loss'])
        batch_accuracies.append(metrics['accuracy'])

    return (
        state,
        jnp.mean(jnp.array(batch_losses)),
        jnp.mean(jnp.array(batch_accuracies))
    )


@partial(jax.jit, static_argnums=(3, 4))
def train_step(
        state: TrainingState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        model: hk.transform,
        classification: bool = False
) -> Tuple[TrainingState, _Metrics]:

    rng_key, next_rng_key = jax.random.split(state.rng_key)

    def loss_fn(params):
        logits = model.apply(params, rng_key, inputs)
        _loss = jnp.mean(cross_entropy_loss(logits, targets))
        _accuracy = jnp.mean(compute_accuracy(logits, targets))
        return _loss, _accuracy

    if not classification:
        targets = inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), gradients = grad_fn(state.params)
    updates, new_opt_state = optimizer.update(gradients, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        rng_key=next_rng_key,
        step=state.step + 1
    )
    metrics = {
        'step': state.step,
        'loss': loss,
        'accuracy': accuracy
    }

    return new_state, metrics


def validate(
        state: TrainingState,
        testloader: DataLoader,
        model: hk.transform,
        classification: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    losses, accuracies = [], []
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        metrics = eval_step(
            state, inputs, targets,
            model, classification
        )
        losses.append(metrics['loss'])
        accuracies.append(metrics['accuracy'])

    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))



@partial(jax.jit, static_argnums=(3, 4))
def eval_step(
        state: TrainingState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        model: hk.transform,
        classification: bool = False
) -> _Metrics:

    rng_key, _ = jax.random.split(state.rng_key, 2)

    if not classification:
        targets = inputs[:, :, 0]

    logits = model.apply(state.params, rng_key, inputs)
    loss = jnp.mean(cross_entropy_loss(logits, targets))
    accuracy = jnp.mean(compute_accuracy(logits, targets))

    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }

    return metrics


def train(dataset: str, seed: int):
    print("[*] Setting Randomness...")
    torch.random.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    classification = "classification" in dataset

    create_dataset_fn = Datasets[dataset]
    trainloader, testloader, n_classes, seq_length, d_input = create_dataset_fn(
        bsz=BATCH_SIZE
    )
    init_data = jnp.array(next(iter(trainloader))[0].numpy())

    @hk.transform
    def model(x):
        neural_net = StackedS4(
            S4D(STATE_SIZE, seq_length),
            D_MODEL,
            N_LAYERS,
            n_classes,
            classification=classification
        )
        return hk.vmap(neural_net, split_rng=False)(x)

    state = init_state(model, rng, init_data)

    for epoch in range(EPOCHS):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss, train_accuracy = train_epoch(
            state, trainloader, model, classification
        )
        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_accuracy = validate(
            state, testloader, model, classification
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_accuracy:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_accuracy:.4f}"
        )

train(dataset='mnist', seed=0)
