import unittest
from core.measurer import PixelsPerMetric


class TestPixelsPerMetric(unittest.TestCase):
    def test_roundtrip(self):
        ppm = PixelsPerMetric(pixels=200.0, metric=50.0)
        # 200 px == 50 mm => 4 px per mm
        self.assertAlmostEqual(ppm.pixels_per_metric, 4.0, places=9)
        self.assertAlmostEqual(ppm.to_metric(8.0), 2.0, places=9)
        self.assertAlmostEqual(ppm.from_metric(2.5), 10.0, places=9)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            PixelsPerMetric(0, 10)
        with self.assertRaises(ValueError):
            PixelsPerMetric(10, 0)


if __name__ == '__main__':
    unittest.main()
