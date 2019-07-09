import contextlib
import typing as tp  # NOQA
import unittest

import numpy
import six

import chainer
from chainer import backend
from chainer import initializers
from chainer.testing import array as array_module
from chainer import utils


class _TestError(AssertionError):

    """Parent class to Chainer test errors."""

    @classmethod
    def check(cls, expr, message):
        if not expr:
            raise cls(message)

    @classmethod
    def fail(cls, message, exc=None):
        if exc is not None:
            utils._raise_from(cls, message, exc)
        raise cls(message)

    @classmethod
    @contextlib.contextmanager
    def raise_if_fail(cls, message, error_types=AssertionError):
        try:
            yield
        except error_types as e:
            cls.fail(message, e)


class FunctionTestError(_TestError):
    """Raised when the target function is implemented incorrectly."""
    pass


class LinkTestError(_TestError):
    """Raised when the target link is implemented incorrectly."""
    pass


class InitializerArgument(object):

    """Class to hold a pair of initializer argument value and actual
    initializer-like.

    This class is meant to be included in the return value from
    :meth:`chainer.testing.LinkTestCase.get_initializers` in
    :class:`chainer.testing.LinkTestCase` if the argument and the actual
    initializer in the link do not directly correspond.
    In that case, the first element should correspond to the argument passed to
    the constructor of the link, and the second element correspond to the
    actual initializer-like object used by the link.
    """

    def __init__(self, argument_value, expected_initializer):
        if expected_initializer is None:
            raise ValueError('Expected initialized cannot be None.')
        initializers._check_is_initializer_like(expected_initializer)

        self.argument_value = argument_value
        self.expected_initializer = expected_initializer


class FunctionTestBase(object):

    backend_config = None
    check_forward_options = None
    check_backward_options = None
    check_double_backward_options = None
    skip_forward_test = False
    skip_backward_test = False
    skip_double_backward_test = False
    dodge_nondifferentiable = False
    contiguous = None

    def __init__(self, *args, **kwargs):
        super(FunctionTestBase, self).__init__(*args, **kwargs)
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}

    def before_test(self, test_name):
        pass

    def forward(self, inputs, device):
        raise NotImplementedError('forward() is not implemented.')

    def forward_expected(self, inputs):
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_inputs(self):
        raise NotImplementedError('generate_inputs() is not implemented.')

    def generate_grad_outputs(self, outputs_template):
        return (numpy.array([[[[[-0.57288873, -0.29788506, -0.24843545, 0.51898718],
               [-0.49249753, -0.88051188, -0.15858935, -0.6599015],
               [-0.51200831, -0.73036814, -0.44067517, -0.64133108],
               [-0.01481274, -0.74166179, 0.69777536, -0.10180384],
               [-0.18441851, 0.82993078, 0.69786048, -0.09297264]],
              [[0.80033565, -0.89548677, 0.45647889, -0.52112365],
               [-0.16415998, 0.25118861, 0.85481554, 0.29262057],
               [0.0570997, -0.82634872, -0.36806324, 0.02658094],
               [-0.84230262, 0.18608826, -0.45148519, 0.69677681],
               [0.62474865, -0.33013779, -0.86770284, 0.98179168]],
              [[0.44201466, 0.11303127, -0.91982031, -0.64594173],
               [-0.58211094, 0.42780966, 0.55538499, -0.83605349],
               [0.19612867, -0.55082911, -0.8137508, 0.70715314],
               [0.94747066, -0.18129289, -0.9617632, -0.65512937],
               [0.72606146, 0.81719613, -0.19496323, 0.47382379]],
              [[-0.66967249, -0.05244569, 0.22894841, 0.62686473],
               [0.49540508, -0.82404941, 0.9015913, -0.58343571],
               [0.27611437, 0.97778851, 0.72046703, 0.59723616],
               [0.24742335, 0.9699654, 0.82666457, 0.61774433],
               [0.52111453, 0.5789156, 0.15938237, -0.87783146]]],
             [[[0.47432569, 0.66716743, -0.76400363, -0.98504454],
               [0.48880142, -0.92792577, 0.30179325, -0.89672023],
               [0.62221336, 0.78228492, 0.24484971, -0.32033888],
               [-0.54916126, 0.55154335, -0.71132892, 0.55365282],
               [0.40025637, -0.52497429, 0.87337887, -0.49099231]],
              [[0.00467373, -0.48013037, 0.46649626, 0.04705672],
               [0.65312022, -0.67780334, -0.36580694, 0.05880155],
               [0.79659408, 0.28774557, -0.14293763, 0.69799769],
               [-0.75784498, -0.34479848, 0.04125214, 0.26577765],
               [-0.82096255, 0.26680973, 0.0412402, -0.27851704]],
              [[-0.3480131, -0.41750696, 0.51542169, 0.58474529],
               [-0.83038008, 0.98628736, 0.60625702, -0.9725076],
               [-0.91647047, 0.99045342, -0.07909115, 0.75326526],
               [-0.25998116, 0.72235167, -0.17732808, -0.66066641],
               [0.3994258, 0.4688834, -0.21801959, -0.49324235]],
              [[0.21084672, -0.99056691, -0.13617717, -0.99912184],
               [-0.57595295, -0.31862435, 0.51519018, -0.31738639],
               [-0.38893867, -0.41463929, -0.80630928, -0.62898439],
               [-0.3204858, -0.8772043, 0.17133164, 0.75059229],
               [0.16891418, 0.28945372, -0.53780627, 0.32307518]]]],
             [[[[0.07748696, -0.46450573, -0.41684186, -0.82503659],
               [-0.23390993, -0.23398358, -0.37380517, -0.72919559],
               [0.71943879, 0.16739683, -0.35677326, -0.99944049],
               [0.41849837, 0.38120511, 0.49531668, 0.64227301],
               [-0.44820809, 0.77409405, -0.78714144, -0.33159694]],
              [[-0.16656028, 0.5239864, -0.6037029, -0.7322076],
               [0.86015332, 0.02132437, 0.07800141, 0.32333809],
               [-0.75669676, 0.91693366, -0.28772381, -0.55460274],
               [0.51093644, 0.06418533, 0.3947728, 0.56368506],
               [-0.87679708, 0.37221545, -0.76213682, 0.20591947]],
              [[-0.87836719, 0.98851788, 0.03740104, -0.83420777],
               [0.21969248, -0.99642915, -0.08743201, 0.9861629],
               [-0.82433724, 0.07615139, -0.75260931, -0.74280965],
               [0.18871161, -0.98334354, -0.08697249, 0.09568729],
               [-0.84707165, 0.36547852, 0.17100099, -0.94585621]],
              [[-0.96832615, -0.02781309, 0.17959639, -0.14480831],
               [0.8213985, -0.77917546, -0.57328159, -0.05056624],
               [0.25292462, -0.33196208, 0.48381472, 0.34306604],
               [0.06550166, 0.33876947, 0.02008327, 0.0945061],
               [0.95860797, -0.58410597, 0.316948, 0.82387859]]],
             [[[-0.31068727, -0.51087368, 0.22985882, 0.56022108],
               [-0.54910862, 0.47769898, -0.86679137, 0.15301202],
               [-0.80621344, 0.4697454, -0.5896886, 0.07740229],
               [-0.26260954, 0.21592896, -0.09272838, 0.66695577],
               [0.59396648, -0.96876419, -0.37223014, 0.76900363]],
              [[0.53733265, -0.07733639, 0.12522766, -0.65647131],
               [-0.50152278, -0.0466333, -0.45405471, -0.32687607],
               [0.43033606, 0.12541339, -0.64357668, -0.21412763],
               [0.69785416, -0.97362012, -0.58536249, 0.78671288],
               [-0.40443432, 0.94519925, 0.53738904, 0.29527199]],
              [[-0.16464908, -0.22645785, 0.65871114, -0.75071335],
               [0.75721818, 0.16671704, 0.74046713, 0.17814854],
               [0.48407817, 0.70155776, 0.51629329, 0.50201118],
               [0.45294064, 0.95396984, -0.80041414, -0.81105053],
               [-0.838287, 0.79126585, -0.99385488, -0.91075659]],
              [[-0.75037426, 0.65934694, 0.12328129, 0.82232845],
               [-0.39328143, -0.57272315, -0.79933745, 0.91013968],
               [-0.19735198, 0.59432393, 0.17559244, 0.36204901],
               [-0.06800383, 0.62853801, 0.56052619, -0.34848008],
               [0.67490572, -0.49171886, 0.67343467, 0.39987543]]]]], dtype=numpy.float32),)
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in inputs_template])
        return grad_grad_inputs

    def check_forward_outputs(self, outputs, expected_outputs):
        assert isinstance(outputs, tuple)
        assert isinstance(expected_outputs, tuple)
        assert all(isinstance(a, chainer.get_array_types()) for a in outputs)
        assert all(
            isinstance(a, chainer.get_array_types()) for a in expected_outputs)
        _check_arrays_equal(
            outputs, expected_outputs, FunctionTestError,
            **self.check_forward_options)

    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))

    def _generate_inputs(self):
        inputs = self.generate_inputs()
        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')
        return inputs

    def _generate_grad_outputs(self, outputs_template):
        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')
        return grad_outputs

    def _generate_grad_grad_inputs(self, inputs_template):
        grad_grad_inputs = self.generate_grad_grad_inputs(inputs_template)
        _check_array_types(
            grad_grad_inputs, backend.CpuDevice(), 'generate_grad_grad_inputs')
        return grad_grad_inputs

    def _forward_expected(self, inputs):
        outputs = self.forward_expected(inputs)
        _check_array_types(
            outputs, backend.CpuDevice(), 'forward_expected')
        return outputs

    def _forward(self, inputs, backend_config):
        assert all(isinstance(a, chainer.Variable) for a in inputs)
        with backend_config:
            outputs = self.forward(inputs, backend_config.device)
        _check_variable_types(
            outputs, backend_config.device, 'forward', FunctionTestError)
        return outputs

    def run_test_forward(self, backend_config):
        # Runs the forward test.

        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        self.backend_config = backend_config
        self.test_name = 'test_forward'
        self.before_test(self.test_name)

        cpu_inputs = self._generate_inputs()
        cpu_inputs = self._to_noncontiguous_as_needed(cpu_inputs)
        inputs_copied = [a.copy() for a in cpu_inputs]

        # Compute expected outputs
        cpu_expected = self._forward_expected(cpu_inputs)

        # Compute actual outputs
        inputs = backend_config.get_array(cpu_inputs)
        inputs = self._to_noncontiguous_as_needed(inputs)
        outputs = self._forward(
            tuple([
                chainer.Variable(a, requires_grad=a.dtype.kind == 'f')
                for a in inputs]),
            backend_config)

        # Check inputs has not changed
        indices = []
        for i in range(len(inputs)):
            try:
                array_module.assert_allclose(
                    inputs_copied[i], inputs[i], atol=0, rtol=0)
            except AssertionError:
                indices.append(i)

        if indices:
            FunctionTestError.fail(
                'Input arrays have been modified during forward.\n'
                'Indices of modified inputs: {}\n'
                'Input array shapes and dtypes: {}\n'.format(
                    ', '.join(str(i) for i in indices),
                    utils._format_array_props(inputs)))

        self.check_forward_outputs(
            tuple([var.array for var in outputs]),
            cpu_expected)

    def run_test_backward(self, backend_config):
        # Runs the backward test.

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.test_name = 'test_backward'
        self.before_test(self.test_name)

        def f(*args):
            return self._forward(args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            inputs = self._to_noncontiguous_as_needed(inputs)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            with FunctionTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check.check_backward(
                    f, inputs, grad_outputs, dtype=numpy.float64,
                    detect_nondifferentiable=self.dodge_nondifferentiable,
                    **self.check_backward_options)

        if self.dodge_nondifferentiable:
            while True:
                try:
                    do_check()
                except gradient_check.NondifferentiableError:
                    continue
                else:
                    break
        else:
            do_check()

    def run_test_double_backward(self, backend_config):
        # Runs the double-backward test.

        if self.skip_double_backward_test:
            raise unittest.SkipTest('skip_double_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.test_name = 'test_double_backward'
        self.before_test(self.test_name)

        def f(*args):
            return self._forward(args, backend_config)

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_grad_inputs = self._generate_grad_grad_inputs(inputs)

            # Drop ggx corresponding to non-differentiable inputs.
            grad_grad_inputs = [
                ggx for ggx in grad_grad_inputs if ggx.dtype.kind == 'f']

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            grad_grad_inputs = backend_config.get_array(grad_grad_inputs)
            inputs = self._to_noncontiguous_as_needed(inputs)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)
            grad_grad_inputs = (
                self._to_noncontiguous_as_needed(grad_grad_inputs))

            with backend_config:
                with FunctionTestError.raise_if_fail(
                        'double backward is not implemented correctly'):
                    gradient_check.check_double_backward(
                        f, inputs, grad_outputs, grad_grad_inputs,
                        dtype=numpy.float64,
                        detect_nondifferentiable=self.dodge_nondifferentiable,
                        **self.check_double_backward_options)

        if self.dodge_nondifferentiable:
            while True:
                try:
                    do_check()
                except gradient_check.NondifferentiableError:
                    continue
                else:
                    break
        else:
            do_check()


class FunctionTestCase(FunctionTestBase, unittest.TestCase):
    """A base class for function test cases.

    Function test cases can inherit from this class to define a set of function
    tests.

    .. rubric:: Required methods

    Each concrete class must at least override the following three methods.

    ``forward(self, inputs, device)``
        Implements the target forward function.
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.

    ``forward_expected(self, inputs)``
        Implements the expectation of the target forward function.
        ``inputs`` is a tuple of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    .. rubric:: Optional methods

    Additionally the concrete class can override the following methods.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'``, ``'test_backward'``, and
        ``'test_double_backward'``.

    ``generate_grad_outputs(self, outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``generate_grad_grad_inputs(self, inputs_template)``
        Returns a tuple of the second order input gradient arrays of type
        :class:`numpy.ndarray`.
        ``input_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``check_forward_outputs(self, outputs, expected_outputs)``
        Implements check logic of forward outputs. Typically additional check
        can be done after calling ``super().check_forward_outputs``.
        ``outputs`` and ``expected_outputs`` are tuples of arrays.
        In case the check fails, ``FunctionTestError`` should be raised.

    .. rubric:: Configurable attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``skip_forward_test`` (bool):
        Whether to skip forward computation test. ``False`` by default.

    ``skip_backward_test`` (bool):
        Whether to skip backward computation test. ``False`` by default.

    ``skip_double_backward_test`` (bool):
        Whether to skip double-backward computation test. ``False`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the inputs returned by ``generate_inputs`` turns
        out to be a non-differentiable point, the test will repeatedly resample
        inputs until a differentiable point will be finally sampled.
        ``False`` by default.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs, output
        gradients, and the second order input gradients). If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. rubric:: Passive attributes

    These attributes are automatically set.

    ``test_name`` (str):
        The name of the test being run. It is one of ``'test_forward'``,
        ``'test_backward'``, and ``'test_double_backward'``.

    ``backend_config`` (:class:`~chainer.testing.BackendConfig`):
        The backend configuration.

    .. note::

       This class assumes :func:`chainer.testing.inject_backend_tests`
       is used together. See the example below.

    .. admonition:: Example

       .. testcode::

          @chainer.testing.inject_backend_tests(
              None,
              [
                  {}, # CPU
                  {'use_cuda': True}, # GPU
             ])
          class TestReLU(chainer.testing.FunctionTestCase):

              # ReLU function has a non-differentiable point around zero, so
              # dodge_nondifferentiable should be set to True.
              dodge_nondifferentiable = True

              def generate_inputs(self):
                  x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
                  return x,

              def forward(self, inputs, device):
                  x, = inputs
                  return F.relu(x),

              def forward_expected(self, inputs):
                  x, = inputs
                  expected = x.copy()
                  expected[expected < 0] = 0
                  return expected,

    .. seealso:: :class:`~chainer.testing.LinkTestCase`

    """

    def test_backward(self, backend_config):
        """Tests backward computation."""
        self.run_test_backward(backend_config)


class _LinkTestBase(object):

    backend_config = None
    contiguous = None

    # List of parameter names represented as strings.
    # I.e. ('gamma', 'beta') for BatchNormalization.
    param_names = ()

    def before_test(self, test_name):
        pass

    def generate_params(self):
        raise NotImplementedError('generate_params is not implemented.')

    def generate_inputs(self):
        raise NotImplementedError('generate_inputs is not implemented.')

    def create_link(self, initializers):
        raise NotImplementedError('create_link is not implemented.')

    def forward(self, link, inputs, device):
        outputs = link(*inputs)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        return outputs

    def check_forward_outputs(self, outputs, expected_outputs):
        assert isinstance(outputs, tuple)
        assert isinstance(expected_outputs, tuple)
        assert all(isinstance(a, chainer.get_array_types()) for a in outputs)
        assert all(
            isinstance(a, chainer.get_array_types()) for a in expected_outputs)
        _check_arrays_equal(
            outputs, expected_outputs, LinkTestError,
            **self.check_forward_options)

    def _generate_params(self):
        params_init = self.generate_params()
        if not isinstance(params_init, (tuple, list)):
            raise TypeError(
                '`generate_params` must return a tuple or a list.')
        for init in params_init:
            _check_generated_initializer(init)
        return params_init

    def _generate_inputs(self):
        inputs = self.generate_inputs()

        _check_array_types(inputs, backend.CpuDevice(), 'generate_inputs')

        return inputs

    def _create_link(self, initializers, backend_config):
        link = self.create_link(initializers)
        if not isinstance(link, chainer.Link):
            raise TypeError(
                '`create_link` must return a chainer.Link object.')

        link.to_device(backend_config.device)

        return link

    def _create_initialized_link(self, inits, backend_config):
        inits = [_get_initializer_argument_value(i) for i in inits]
        link = self._create_link(inits, backend_config)

        # Generate inputs and compute a forward pass to initialize the
        # parameters.
        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        inputs_xp = self._to_noncontiguous_as_needed(inputs_xp)
        input_vars = [chainer.Variable(i) for i in inputs_xp]
        output_vars = self._forward(link, input_vars, backend_config)
        outputs_xp = [v.array for v in output_vars]

        link.cleargrads()

        return link, inputs_xp, outputs_xp

    def _forward(self, link, inputs, backend_config):
        assert all(isinstance(x, chainer.Variable) for x in inputs)

        with backend_config:
            outputs = self.forward(link, inputs, backend_config.device)
        _check_variable_types(
            outputs, backend_config.device, 'forward', LinkTestError)

        return outputs

    def _to_noncontiguous_as_needed(self, contig_arrays):
        if self.contiguous is None:
            # non-contiguous
            return array_module._as_noncontiguous_array(contig_arrays)
        if self.contiguous == 'C':
            # C-contiguous
            return contig_arrays
        assert False, (
            'Invalid value of `contiguous`: {}'.format(self.contiguous))


class LinkTestCase(_LinkTestBase, unittest.TestCase):

    """A base class for link forward and backward test cases.

    Link test cases can inherit from this class to define a set of link tests
    for forward and backward computations.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_params(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link``.

    ``create_link(self, initializers)``
        Returns a link. The link should be initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    ``forward(self, link, inputs, device)``
        Implements the target forward function.
        ``link`` is a link created by ``create_link`` and
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.
        A default implementation is provided for links that only takes the
        inputs defined in ``generate_inputs`` (wrapped in
        :class:`~chainer.Variable`\\ s) and returns nothing but output
        :class:`~chainer.Variable`\\ s in its forward computation.

    .. rubric:: Optional methods

    Each concrete class may override the following methods depending on the
    skip flags  ``skip_forward_test`` and  ``skip_backward_test``.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is one of ``'test_forward'`` and ``'test_backward'``.

    ``forward_expected(self, link, inputs)``
        Implements the expectation of the target forward function.
        ``link`` is the initialized link that was used to compute the actual
        forward which the results of this method will be compared against.
        The link is guaranteed to reside on the CPU.
        ``inputs`` is a tuple of :class:`numpy.ndarray`\\ s.
        This method is expected to return the output
        :class:`numpy.ndarray`\\ s.
        This method must be implemented if either ``skip_forward_test`` or
        ``skip_backward_test`` is ``False`` in which case forward or backward
        tests are executed.

    ``generate_grad_outputs(self, outputs_template)``
        Returns a tuple of output gradient arrays of type
        :class:`numpy.ndarray`.
        ``outputs_template`` is a tuple of template arrays. The returned arrays
        are expected to have the same shapes and dtypes as the template arrays.

    ``check_forward_outputs(self, outputs, expected_outputs)``
        Implements check logic of forward outputs. Typically additional check
        can be done after calling ``super().check_forward_outputs``.
        ``outputs`` and ``expected_outputs`` are tuples of arrays.
        In case the check fails, ``LinkTestError`` should be raised.

    .. rubric:: Attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``param_names`` (tuple of str):
        A tuple of strings with all the names of the parameters that should be
        tested. E.g. ``('gamma', 'beta')`` for the batch normalization link.
        ``()`` by default.

    ``skip_forward_test`` (bool):
        Whether to skip forward computation test. ``False`` by default.

    ``skip_backward_test`` (bool):
        Whether to skip backward computation test. ``False`` by default.

    ``dodge_nondifferentiable`` (bool):
        Enable non-differentiable point detection in numerical gradient
        calculation. If the data returned by
        ``generate_params``, ``create_link`` and ``generate_inputs`` turns out
        to be a non-differentiable point, the test will repeatedly resample
        those until a differentiable point will be finally sampled. ``False``
        by default.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs,
        parameters and gradients. If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. note::

        This class assumes :func:`chainer.testing.inject_backend_tests`
        is used together. See the example below.

    .. note::

        When implementing :class:`~chainer.testing.LinkTestCase` and
        :class:`~chainer.testing.LinkInitializersTestCase` to test both
        forward/backward and initializers, it is often convenient to refactor
        out common logic in a separate class.

    .. admonition:: Example

        .. testcode::

            @chainer.testing.inject_backend_tests(
              None,
              [
                  {}, # CPU
                  {'use_cuda': True}, # GPU
             ])
            class TestLinear(chainer.testing.LinkTestCase):

                param_names = ('W', 'b')

                def generate_params(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
                    return initialW, initial_bias

                def generate_inputs(self):
                    x = numpy.random.uniform(
                        -1, 1, (1, 2)).astype(numpy.float32)
                    return x,

                def create_link(self, initializers):
                    initialW, initial_bias = initializers
                    link = chainer.links.Linear(
                        2, 3, initialW=initialW, initial_bias=initial_bias)
                    return link

                def forward(self, link, inputs, device):
                    x, = inputs
                    return link(x),

                def forward_expected(self, link, inputs):
                    W = link.W.array
                    b = link.b.array
                    x, = inputs
                    expected = x.dot(W.T) + b
                    return expected,

    .. seealso::
        :class:`~chainer.testing.LinkInitializersTestCase`
        :class:`~chainer.testing.FunctionTestCase`

    """

    check_forward_options = None
    check_backward_options = None
    skip_forward_test = False
    skip_backward_test = False
    dodge_nondifferentiable = False

    def __init__(self, *args, **kwargs):
        self.check_forward_options = {}
        self.check_backward_options = {}

        super(LinkTestCase, self).__init__(*args, **kwargs)

    def forward_expected(self, link, inputs):
        raise NotImplementedError('forward_expected() is not implemented.')

    def generate_grad_outputs(self, outputs_template):
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])
        return grad_outputs

    def test_forward(self, backend_config):
        """Tests forward computation."""

        if self.skip_forward_test:
            raise unittest.SkipTest('skip_forward_test is set')

        self.backend_config = backend_config
        self.before_test('test_forward')

        inits = self._generate_params()
        link = self._create_link(inits, backend_config)

        inputs_np = self._generate_inputs()
        inputs_xp = backend_config.get_array(inputs_np)
        inputs_xp = self._to_noncontiguous_as_needed(inputs_xp)
        input_vars = tuple([chainer.Variable(i) for i in inputs_xp])
        # Compute forward of the link and initialize its parameters.
        output_vars = self._forward(link, input_vars, backend_config)
        outputs_xp = [v.array for v in output_vars]

        # Expected outputs are computed on the CPU so the link must be
        # transferred.
        link.to_device(backend.CpuDevice())

        expected_outputs_np = self._forward_expected(link, inputs_np)

        self.check_forward_outputs(
            tuple(outputs_xp), expected_outputs_np)

    def test_backward(self, backend_config):
        """Tests backward computation."""

        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        self.backend_config = backend_config
        self.before_test('test_backward')

        # avoid cyclic import
        from chainer import gradient_check

        def do_check():
            # Generate an initialized temporary link that is already forward
            # propagated. This link is only used to generate necessary data,
            # i.e. inputs, outputs and parameters for the later gradient check
            # and the link itself will be discarded.
            inits = self._generate_params()
            link, inputs, outputs = self._create_initialized_link(
                inits, backend_config)

            # Extract the parameter ndarrays from the initialized link.
            params = _get_link_params(link, self.param_names)
            params = [p.array for p in params]

            # Prepare inputs, outputs and upstream gradients for the gradient
            # check.
            cpu_device = backend.CpuDevice()
            outputs = [cpu_device.send(output) for output in outputs]
            grad_outputs = self._generate_grad_outputs(outputs)
            grad_outputs = backend_config.get_array(grad_outputs)

            inputs = self._to_noncontiguous_as_needed(inputs)
            params = self._to_noncontiguous_as_needed(params)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)

            # Create the link used for the actual forward propagation in the
            # gradient check.
            forward_link, _, _ = self._create_initialized_link(
                inits, backend_config)

            def forward(inputs, ps):

                # Use generated parameters.
                with forward_link.init_scope():
                    for param_name, p in zip(self.param_names, ps):
                        setattr(forward_link, param_name, p)

                return self._forward(forward_link, inputs, backend_config)

            with LinkTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check._check_backward_with_params(
                    forward, inputs, grad_outputs, params=params,
                    dtype=numpy.float64,
                    detect_nondifferentiable=self.dodge_nondifferentiable,
                    **self.check_backward_options)

        if self.dodge_nondifferentiable:
            while True:
                try:
                    do_check()
                except gradient_check.NondifferentiableError:
                    continue
                else:
                    break
        else:
            do_check()

    def _forward_expected(self, link, inputs):
        assert all(isinstance(x, numpy.ndarray) for x in inputs)

        outputs = self.forward_expected(link, inputs)
        _check_array_types(inputs, backend.CpuDevice(), 'test_forward')

        return outputs

    def _generate_grad_outputs(self, outputs_template):
        assert all(isinstance(x, numpy.ndarray) for x in outputs_template)

        grad_outputs = self.generate_grad_outputs(outputs_template)
        _check_array_types(
            grad_outputs, backend.CpuDevice(), 'generate_grad_outputs')

        return grad_outputs


class LinkInitializersTestCase(_LinkTestBase, unittest.TestCase):

    """A base class for link parameter initializer test cases.

    Link test cases can inherit from this class to define a set of link tests
    for parameter initialization.

    .. rubric:: Required methods

    Each concrete class must at least override the following methods.

    ``generate_params(self)``
        Returns a tuple of initializers-likes. The tuple should contain an
        initializer-like for each initializer-like argument, i.e. the
        parameters to the link constructor. These will be passed to
        ``create_link``.

    ``create_link(self, initializers)``
        Returns a link. The link should be initialized with the given
        initializer-likes ``initializers``. ``initializers`` is a tuple of
        same length as the number of parameters.

    ``generate_inputs(self)``
        Returns a tuple of input arrays of type :class:`numpy.ndarray`.

    ``forward(self, link, inputs, device)``
        Implements the target forward function.
        ``link`` is a link created by ``create_link`` and
        ``inputs`` is a tuple of :class:`~chainer.Variable`\\ s.
        This method is expected to return the output
        :class:`~chainer.Variable`\\ s with the same array types as the inputs.
        ``device`` is the device corresponding to the input arrays.
        A default implementation is provided for links that only takes the
        inputs defined in ``generate_inputs`` (wrapped in
        :class:`~chainer.Variable`\\ s) and returns nothing but output
        :class:`~chainer.Variable`\\ s in its forward computation.

    ``get_initializers(self)``
        Returns a tuple with the same length as the number of initializers that
        the constructor of the link accepts. Each element in the tuple is a
        container itself, listing all initializers-likes that should be tested.
        Each initializer-like in the tuple is tested one at a time by being
        passed to ``create_link``. When the length of the tuple is greater than
        one (i.e. if the link accepts multiple initializers), the ones not
        being tested are replaced by the ones returned by `generate_params`.
        Initializer-likes returned here should be deterministic since test will
        invoke them multiple times to test the correctness.

        For testing initializer arguments that can be non-initializer values
        such as ``None``, one can use the ``InitializerArgument``, defining a
        pair of the link constructor argument and actual initializer-like used
        by the link.
        This method must be implemented if ``skip_initializers_test`` is
        ``False`` in which case the initializers test is executed.

    .. rubric:: Optional methods

    Each concrete class may override the following methods.

    ``before_test(self, test_name)``
        A callback method called before each test.
        Typically a skip logic is implemented by conditionally raising
        :class:`unittest.SkipTest`.
        ``test_name`` is always of ``'test_initializers'``.

    .. rubric:: Attributes

    The concrete class can override the following attributes to control the
    behavior of the tests.

    ``param_names`` (list of str):
        A list of strings with all the names of the parameters that should be
        tested. E.g. ``['gamma', 'beta']`` for the batch normalization link.
        ``[]`` by default.

    ``contiguous`` (None or 'C'):
        Specifies the contiguousness of incoming arrays (i.e. inputs,
        parameters and gradients. If ``None``, the
        arrays will be non-contiguous as long as possible. If ``'C'``, the
        arrays will be C-contiguous. ``None`` by default.

    .. note::

        This class assumes :func:`chainer.testing.inject_backend_tests`
        is used together. See the example below.

    .. note::

        When implementing :class:`~chainer.testing.LinkTestCase` and
        :class:`~chainer.testing.LinkInitializersTestCase` to test both
        forward/backward and initializers, it is often convenient to refactor
        out common logic in a separate class.

    .. admonition:: Example

        .. testcode::

            @chainer.testing.inject_backend_tests(
              None,
              [
                  {}, # CPU
                  {'use_cuda': True}, # GPU
             ])
            class TestLinear(chainer.testing.LinkInitializersTestCase):

                param_names = ['W', 'b']

                def generate_params(self):
                    initialW = numpy.random.uniform(
                        -1, 1, (3, 2)).astype(numpy.float32)
                    initial_bias = numpy.random.uniform(
                        -1, 1, (3,)).astype(numpy.float32)
                    return initialW, initial_bias

                def generate_inputs(self):
                    x = numpy.random.uniform(
                        -1, 1, (1, 2)).astype(numpy.float32)
                    return x,

                def create_link(self, initializers):
                    initialW, initial_bias = initializers
                    link = chainer.links.Linear(
                        2, 3, initialW=initialW, initial_bias=initial_bias)
                    return link

                def forward(self, link, inputs, device):
                    x, = inputs
                    return link(x),

                def get_initializers(self):
                    initialW = [initializers.Constant(1), 2]
                    initial_bias = [initializers.Constant(2), 3,
                        chainer.testing.link.InitializerArgument(None, 0)]
                    return initialW, initial_bias

    .. seealso::
        :class:`~chainer.testing.LinkTestCase`
        :class:`~chainer.testing.FunctionTestCase`

    """

    check_initializers_options = None

    def __init__(self, *args, **kwargs):
        self.check_initializers_options = {}

        super(LinkInitializersTestCase, self).__init__(*args, **kwargs)

    def get_initializers(self):
        raise NotImplementedError('get_initializers is not implemented.')

    def test_initializers(self, backend_config):
        """Tests that the parameters of a links are correctly initialized."""

        self.backend_config = backend_config
        self.before_test('test_initializers')

        params_inits = self._get_initializers()

        # TODO(hvy): Reduce the number of loop iterations by checking
        # multiple parameters simultaneously.
        for i_param, param_inits in enumerate(params_inits):
            # When testing an initializer for a particular parameter, other
            # initializers are picked from generate_params.
            inits = self._generate_params()
            inits = list(inits)

            for init in param_inits:
                inits[i_param] = init
                self._test_single_initializer(i_param, inits, backend_config)

    def _get_initializers(self):
        params_inits = self.get_initializers()
        if not isinstance(params_inits, (tuple, list)):
            raise TypeError(
                '`get_initializers` must return a tuple or a list.')
        for param_inits in params_inits:
            if not isinstance(param_inits, (tuple, list)):
                raise TypeError(
                    '`get_initializers` must return a tuple or a list of '
                    'tuples or lists.')
            for init in param_inits:
                _check_generated_initializer(init)
        return params_inits

    def _test_single_initializer(self, i_param, inits, backend_config):
        # Given a set of initializer constructor arguments for the link, create
        # and initialize a link with those arguments. `i_param` holds the index
        # of the argument that should be tested among these.
        inits_orig = inits
        inits = [_get_initializer_argument_value(i) for i in inits]
        link, _, _ = self._create_initialized_link(inits, backend_config)

        # Extract the parameters from the initialized link.
        params = _get_link_params(link, self.param_names)

        # Convert the parameter of interest into a NumPy ndarray.
        cpu_device = backend.CpuDevice()
        param = params[i_param]
        param_xp = param.array
        param_np = cpu_device.send(param_xp)

        # The expected values of the parameter is decided by the given
        # initializer. If the initializer is `None`, it should have been
        # wrapped in a InitializerArgument along with the expected initializer
        # that the link should default to in case of `None`.
        #
        # Note that for this to work, the expected parameter must be inferred
        # deterministically.
        expected_init = _get_expected_initializer(inits_orig[i_param])
        expected_np = numpy.empty_like(param_np)
        expected_init(expected_np)

        # Compare the values of the expected and actual parameter.
        _check_arrays_equal(
            (expected_np,), (param_np,), LinkTestError,
            **self.check_initializers_options)


def _check_generated_initializer(init):
    if isinstance(init, InitializerArgument):
        init = init.expected_initializer
    elif init is None:
        raise ValueError(
            'A None initializer must be wrapped in a InitializerArgument '
            'along with the expected initializer fallen back to.')
    initializers._check_is_initializer_like(init)


def _get_initializer_argument_value(init):
    # Returns the initializer that should be passed to the link constructor.

    if isinstance(init, InitializerArgument):
        return init.argument_value
    return init


def _get_expected_initializer(init):
    # Returns the expected initializer for the given initializer.

    if isinstance(init, InitializerArgument):
        init = init.expected_initializer

    assert init is not None

    if not isinstance(init, chainer.Initializer):
        init = chainer.initializers._get_initializer(init)
    return init


def _get_link_params(link, param_names):
    params = []
    for name in param_names:
        param = getattr(link, name, None)
        if param is None:
            raise LinkTestError.fail(
                'Link does not have a parameter named \'{}\'.'.format(name))
        params.append(param)
    return params


def _check_array_types(arrays, device, func_name):
    if not isinstance(arrays, tuple):
        raise TypeError(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(arrays)))
    if not all(isinstance(a, device.supported_array_types) for a in arrays):
        raise TypeError(
            '{}() must return a tuple of arrays supported by device {}.\n'
            'Actual: {}'.format(
                func_name, device, tuple([type(a) for a in arrays])))


def _check_variable_types(vars, device, func_name, test_error_cls):
    assert issubclass(test_error_cls, _TestError)

    if not isinstance(vars, tuple):
        test_error_cls.fail(
            '`{}()` must return a tuple, '
            'not {}.'.format(func_name, type(vars)))
    if not all(isinstance(a, chainer.Variable) for a in vars):
        test_error_cls.fail(
            '{}() must return a tuple of Variables.\n'
            'Actual: {}'.format(
                func_name, ', '.join(str(type(a)) for a in vars)))
    if not all(isinstance(a.array, device.supported_array_types)
               for a in vars):
        test_error_cls.fail(
            '{}() must return a tuple of Variables of arrays supported by '
            'device {}.\n'
            'Actual: {}'.format(
                func_name, device,
                ', '.join(str(type(a.array)) for a in vars)))


def _check_arrays_equal(
        actual_arrays, expected_arrays, test_error_cls, **opts):
    # `opts` is passed through to `testing.assert_all_close`.
    # Check all outputs are equal to expected values
    assert issubclass(test_error_cls, _TestError)

    message = None
    detail_message = None
    while True:
        # Check number of arrays
        if len(actual_arrays) != len(expected_arrays):
            message = (
                'Number of outputs ({}, {}) does not match'.format(
                    len(actual_arrays), len(expected_arrays)))
            break

        # Check dtypes and shapes
        dtypes_match = all([
            y.dtype == ye.dtype
            for y, ye in zip(actual_arrays, expected_arrays)])
        shapes_match = all([
            y.shape == ye.shape
            for y, ye in zip(actual_arrays, expected_arrays)])
        if not (shapes_match and dtypes_match):
            message = 'Shapes and/or dtypes do not match'
            break

        # Check values
        errors = []
        for i, (actual, expected) in (
                enumerate(zip(actual_arrays, expected_arrays))):
            try:
                array_module.assert_allclose(actual, expected, **opts)
            except AssertionError as e:
                errors.append((i, e))
        if errors:
            message = (
                'Outputs do not match the expected values.\n'
                'Indices of outputs that do not match: {}'.format(
                    ', '.join(str(i) for i, e in errors)))
            f = six.StringIO()
            for i, e in errors:
                f.write('Error details of output [{}]:\n'.format(i))
                f.write(str(e))
                f.write('\n')
            detail_message = f.getvalue()
            break
        break

    if message is not None:
        msg = (
            '{}\n'
            'Expected shapes and dtypes: {}\n'
            'Actual shapes and dtypes:   {}\n'.format(
                message,
                utils._format_array_props(expected_arrays),
                utils._format_array_props(actual_arrays)))
        if detail_message is not None:
            msg += '\n\n' + detail_message
        test_error_cls.fail(msg)
