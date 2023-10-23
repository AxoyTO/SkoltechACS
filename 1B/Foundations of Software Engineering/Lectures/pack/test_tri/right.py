import unittest
from pack.tri.right import TriRight

class TestTriRight(unittest.TestCase):
    def test_valid1(self):
        self.assertEqual(TriRight(3.0, 4.0, 5.0).c, 5.0)
        self.assertEqual(TriRight(0.3, 0.4).c, 0.5)
        self.assertEqual(TriRight(3.e+300, 4.0+300).c, 5.0e+300)
        self.assertEqual(type(TriRight(3,4).c), int)

    def test_invalid1(self):
        with self.assertRaises(ValueError):
            TriRight("3.0", "4.0")

    def test_border1(self):
        self.assertEqual(TriRight(0.0, -4.0).c, 4.0)