import unittest

from number_theoretic_transform import ntt, intt, ntt_convolve


class TestRoundTrip(unittest.TestCase):
    def test_ntt_then_intt_is_identity(self):
        a = [1, 2, 3, 4]
        self.assertEqual(intt(ntt(a)), [1, 2, 3, 4])

    def test_round_trip_with_zeros(self):
        a = [0, 0, 0, 0]
        self.assertEqual(intt(ntt(a)), a)

    def test_single_element(self):
        self.assertEqual(ntt([7]), [7])
        self.assertEqual(intt([7]), [7])

    def test_negative_inputs_reduced(self):
        a = [-1, -2, 3, -4]
        out = intt(ntt(a))
        self.assertEqual(out, [p - 1, p - 2, 3, p - 4] if False else [x % p for x in a])


class TestConvolution(unittest.TestCase):
    def test_simple_polynomial_multiply(self):
        # (1 + x) * (1 + x) = 1 + 2x + x^2
        self.assertEqual(ntt_convolve([1, 1], [1, 1]), [1, 2, 1, 0])

    def test_convolve_with_constant(self):
        self.assertEqual(ntt_convolve([3], [1, 2, 3]), [3, 6, 9, 0])

    def test_longer_vectors(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        self.assertEqual(ntt_convolve(a, b), [5, 16, 34, 60, 61, 52, 32, 0])

    def test_asymmetric_lengths(self):
        self.assertEqual(ntt_convolve([1, 2, 3], [4, 5]), [4, 13, 22, 15])

    def test_result_modulus(self):
        out = ntt_convolve([1], [1])
        self.assertTrue(all(0 <= c < p for c in out))


class TestValidation(unittest.TestCase):
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            ntt([])
        with self.assertRaises(ValueError):
            intt([])
        with self.assertRaises(ValueError):
            ntt_convolve([], [1])
        with self.assertRaises(ValueError):
            ntt_convolve([1], [])

    def test_non_power_of_two_length_rejected(self):
        with self.assertRaises(ValueError):
            ntt([1, 2, 3])
        with self.assertRaises(ValueError):
            intt([1, 2, 3])

    def test_non_int_input_rejected(self):
        with self.assertRaises(TypeError):
            ntt([1, 2.0, 3, 4])
        with self.assertRaises(TypeError):
            ntt_convolve([1, 2], [1, 2.0])

    def test_non_list_input_rejected(self):
        with self.assertRaises(TypeError):
            ntt("abcd")
        with self.assertRaises(TypeError):
            ntt_convolve(123, [1, 2])


p = 998244353


if __name__ == "__main__":
    unittest.main()
