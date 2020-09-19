import unittest
from typing import Set

from src import app


class TestApp(unittest.TestCase):
    def test_almost_equal(self):
        self.assertTrue(app.almost_equal(1e-5, 1e-5))

    def test_random_cities(self):
        cities = app.random_cities(app.Config(num_cities=10, country="US"))
        self.assertEqual(len(cities), 10)
        unique: Set[str] = set()
        for city in cities:
            self.assertEqual(city.country, "US")
            unique.add(city.name)
        self.assertEqual(len(unique), 10)


if __name__ == '__main__':
    unittest.main()
