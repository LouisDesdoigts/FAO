import jax
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import optax
import equinox as eqx
import zodiax as zdx
from tqdm.notebook import tqdm


hard_tanh = jax.nn.hard_tanh
sigmoid = jax.nn.sigmoid
relu = jax.nn.relu
identity = lambda x: x


def sample_normal(n, size, shape, mu=0, sigma=1, key=0):
    return mu + (sigma * jr.normal(jr.PRNGKey(key), (n, size) + shape))


def poisson(fmodel, data):
    return -jax.scipy.stats.poisson.logpmf(data, fmodel.model()).mean()


def chi2(fmodel, data):
    return -jax.scipy.stats.chi2.logpdf(data, fmodel.model()).mean()


def MSE(fmodel, data):
    return np.square(data - fmodel.model()).mean()


def MAE(fmodel, data):
    return np.abs(data - fmodel.model()).mean()


def MAPE(fmodel, data):
    return np.abs(data - fmodel.model()).mean() / data.mean()


def MSPE(fmodel, data):
    return np.square(data - fmodel.model()).mean() / data.mean()


class SimpleLinear(eqx.Module):
    """Match the Linear layer API for simplicity"""

    weight: jax.Array
    bias: jax.Array

    # Add polynomial relationship to handle parameter scales?
    def __init__(self, shape):
        self.weight = np.ones(shape, float)
        self.bias = np.zeros(shape, float)

    def __call__(self, x):
        return x * self.weight + self.bias


class OptNet(zdx.Base):
    use_loss: bool = eqx.field(static=True)
    params: list = eqx.field(static=True)
    use_ratio: bool
    loss_type: str
    fmodel: object
    optimiser: object

    def __init__(
        self,
        fmodel,
        params,
        key=None,
        depth=0,
        width=None,
        activation=relu,
        final_activation=identity,
        use_bias=True,
        use_final_bias=True,
        use_loss=False,
        loss_type="MSE",
        use_ratio=False,
    ):
        """Constructs an MLP as an optimiser."""
        self.fmodel = fmodel
        self.params = params
        self.use_loss = use_loss
        self.use_ratio = use_ratio

        if loss_type not in ("MSE", "MAE", "MAPE", "MSPE", "chi2", "poisson"):
            raise ValueError(f"Loss {loss_type} not recognised")
        self.loss_type = loss_type

        if width is None:
            width = len(self.format_input(fmodel))

        if use_loss:
            out_shape = len(fmodel.get(params))
            in_shape = out_shape + 1
        else:
            out_shape = len(fmodel.get(params))
            in_shape = out_shape

        if key is None:
            rand = True
            key = 0  # Set to zero, but the values get overwritten anyway
        else:
            rand = False

        # Initialise the optimiser
        optimiser = eqx.nn.MLP(
            in_size=in_shape,
            width_size=width,
            out_size=out_shape,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=jr.PRNGKey(key),
        )

        # Create global and local learning rate layers
        local_layer = SimpleLinear(in_shape)
        new_layers = [local_layer] + list(optimiser.layers)

        # Initialise values to zero
        if not rand:
            new_layers = [
                eqx.tree_at(
                    lambda tree: (tree.weight, tree.bias),
                    layer,
                    (layer.weight / layer.weight, 0 * layer.bias),
                )
                for layer in new_layers
            ]

        # Set new layers
        optimiser = eqx.tree_at(
            lambda tree: tree.layers, optimiser, new_layers
        )
        self.optimiser = optimiser

    def __getattr__(self, name):
        if hasattr(self.optimiser, name):
            return getattr(self.optimiser, name)
        raise AttributeError(f"OptNet has no attribute {name}")

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @property
    def image_loss(self):
        if self.loss_type == "MSE":
            return MSE
        elif self.loss_type == "MAE":
            return MAE
        elif self.loss_type == "MAPE":
            return MAPE
        elif self.loss_type == "MSPE":
            return MSPE
        elif self.loss_type == "chi2":
            return chi2
        elif self.loss_type == "poisson":
            return poisson
        else:
            raise ValueError(f"Loss {self.loss_type} not recognised")

    def model_value(self, values):
        """Models a single input value and returns the PSF"""
        return self.fmodel.set(self.params, values).model()

    def model_values(self, values):
        """Models a batch of input values and returns the PSFs"""
        return jax.vmap(self.model_value)(values)

    def eval_grads(self, values, data):
        """Returns the gradients of the loss function"""
        loss_fn = lambda optics: self.image_loss(optics, data)
        return eqx.filter_grad(loss_fn)(self.fmodel.set(self.params, values))

    def eval_loss_grads(self, values, data):
        """Returns the loss and gradients of the loss function"""
        loss_fn = lambda optics: self.image_loss(optics, data)
        return eqx.filter_value_and_grad(loss_fn)(
            self.fmodel.set(self.params, values)
        )

    def format_input(self, tree, loss=None):
        """Formats the loss and grads pytree into a flat vector"""
        vals_list = jtu.tree_flatten(tree.get(self.params))[0]
        flat_vals = jtu.tree_map(lambda x: x.flatten(), vals_list)
        if loss is None:
            return np.concatenate(flat_vals)
        return np.concatenate(flat_vals + [np.array([loss])])

    def predict(self, values, data):
        """Return the predicted values"""
        if self.use_loss:
            loss, grads = self.eval_loss_grads(values, data)
            in_vec = self.format_input(grads, loss)
        else:
            grads = self.eval_grads(values, data)
            in_vec = self.format_input(grads)
        return values - self.optimiser(in_vec)

    # TODO: Test as ratio of initial and final loss
    def unsupervised_loss(self, values, data):
        """Returns the unsupervised loss"""
        initial_loss = self.image_loss(
            self.fmodel.set(self.params, values), data
        )
        predicted_values = self.predict(values, data)
        predicted_fmodel = self.fmodel.set(self.params, predicted_values)
        final_loss = self.image_loss(predicted_fmodel, data)
        if self.use_ratio:
            return final_loss / initial_loss
        return final_loss - initial_loss

    # TODO: Distance ratio?
    def supervised_loss(self, values, data, truth):
        """Returns the supervised loss"""
        init_dist = np.linalg.norm(values - truth)
        final_dist = np.linalg.norm(self.predict(values, data) - truth)
        if self.use_ratio:
            return final_dist / init_dist
        # return init_dist - final_dist
        return final_dist - init_dist


class Trainer(zdx.Base):
    model: object
    opt_state: object
    optim: object
    losses: jax.Array
    filter_spec: object = eqx.field(static=True)

    def __init__(self, model, optim):
        self.model = model
        self.losses = None
        self.optim = optim
        filtered_optimiser = eqx.filter(self.model, eqx.is_array)
        self.opt_state = self.optim.init(filtered_optimiser)

        # Make a filter to only train on the optimiser parameters
        false_tree = jtu.tree_map(lambda _: False, model)
        true_tree = jtu.tree_map(lambda _: True, model.optimiser)
        self.filter_spec = eqx.tree_at(
            lambda tree: tree.optimiser, false_tree, true_tree
        )

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def training_loss(self, dynamic, static, values, data, truth, supervised):
        """Returns the loss of the optimiser model"""
        model = eqx.combine(dynamic, static)
        if supervised:
            return model.supervised_loss(values, data, truth)
        return model.unsupervised_loss(values, data)

    def batched_loss(self, dynamic, static, values, data, truth, supervised):
        """Returns the loss of the optimiser model"""
        batched_fn = eqx.filter_vmap(
            self.training_loss, in_axes=(None, None, 0, None, None, None)
        )
        return batched_fn(
            dynamic, static, values, data, truth, supervised
        ).mean()

    def make_step(self, model, opt_state, *args):
        """Hig-level jit-ed function"""
        # Calculate loss and grads
        loss_grads = lambda *args: eqx.filter_value_and_grad(
            self.batched_loss
        )(*args)
        loss, grads = loss_grads(
            *eqx.partition(model, self.filter_spec), *args
        )

        # Update model and return
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train_single(
        self, samples, data=None, truth=None, kth=50, verbose=True
    ):
        """Assumes 'samples' has shape (n epochs, batch size, n params)"""
        # Get model, opt_state and data
        model, opt_state = self.model, self.opt_state

        if data is None and truth is None:
            raise ValueError(
                "Either data or truth must be passed. Passing data results "
                "in unsupervised training, passing truth results in supervised training"
            )

        # Truth already exists, make data
        if truth is not None:
            data = model.model_value(truth)
            supervised = True
            print("Supervised training")

        # Data already exists, make truth
        else:
            truth = np.zeros(self.model.fmodel.get(self.model.params).shape)
            supervised = False
            print("Unsupervised training")

        # Bind static parameters and compile the step function
        step_fn = eqx.filter_jit(
            lambda model, opt_state, sample: self.make_step(
                model, opt_state, sample, data, truth, supervised
            )
        )
        _ = step_fn(model, opt_state, samples[0])

        # Training loop
        losses = []
        with tqdm(range(len(samples)), desc="Loss") as t:
            for i in t:
                loss, model, opt_state = step_fn(model, opt_state, samples[i])
                if verbose and i % kth == 0:
                    print(f"Epoch: {i} \t Loss {loss.mean():.5}")
                losses.append(loss.mean())
                t.set_description(
                    f"Loss {loss.mean():.5}"
                )  # update the progress bar
        if verbose:
            print(f"Epoch: {i+1} \tLoss {loss.mean():.5}")

        # Update parameters
        if self.losses is not None:
            losses = np.concatenate((self.losses, np.array(losses)))

        return eqx.tree_at(
            lambda tree: (tree.model, tree.opt_state, tree.losses),
            self,
            (model, opt_state, np.array(losses)),
            is_leaf=lambda x: x is None,
        )

    def train(
        self, truths, init_ranges, n_samples, batch_size, kth=50, verbose=True
    ):
        """
        True values
            The value at which to generate the data. Automatically generate data.
        init ranges:
            The ranges over which to generate the initial values away from the truth.
            Automatically generate samples.
        n_samples:
            How many times to sample each data image.
        batch_size:
            How many sampled to train on at once.
        supervised: bool
            Whether to train supervised or unsupervised.
        kth:
            How often to print the loss.
        verbose: bool
            Whether to print the loss.
        """
        raise NotImplementedError
