import math
import random
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Tuple, NamedTuple, Dict, Optional

import faker
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from faker import Faker

st.beta_set_page_config("TSP - GA", "https://www.flaticon.com/svg/static/icons/svg/252/252025.svg")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.header("Parameters")

fake = Faker()

R = 6371e3
EPSILON = 1e-6


@dataclass(frozen=True)
class Config(object):
    num_cities: int = 10
    population_size: int = 100
    elite_size: int = 20
    mutation_rate: float = 0.01
    num_generations: int = 150
    country: str = "US"
    max_trials: int = 1_000


def almost_equal(n1: float, n2: float) -> bool:
    return abs(n1 - n2) < EPSILON


@dataclass(frozen=True)
class Coordinates(object):
    latitude: float
    longitude: float

    def __eq__(self, other):
        return almost_equal(self.latitude, other.latitude) and almost_equal(self.longitude, other.longitude)


@dataclass(frozen=True)
class City(object):
    index: int
    coordinates: Coordinates
    name: str
    country: str
    timezone: str

    def __eq__(self, other):
        return self.coordinates == other.coordinates


@dataclass(frozen=True)
class Route(object):
    route: List[City]

    @property
    def route_distance(self) -> float:
        d: float = 0
        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = self.route[i + 1] if i + 1 < len(self.route) else self.route[0]
            d += distance(from_city.coordinates, to_city.coordinates)
        return d

    @property
    def route_fitness(self) -> float:
        return 1 / self.route_distance

    def __len__(self):
        return len(self.route)

    def __getitem__(self, item):
        return self.route[item]

    def __setitem__(self, key, value):
        self.route[key] = value


@lru_cache(maxsize=2048)
def distance(a: Coordinates, b: Coordinates) -> float:
    """Returns the (haversine) distance between 2 points in Kilometers"""
    phi1 = a.latitude * math.pi / 180
    phi2 = b.latitude * math.pi / 180
    delta_phi = (b.latitude - a.latitude) * math.pi / 180
    delta_lambda = (b.longitude - a.longitude) * math.pi / 180

    a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(phi1) * math.cos(
        phi2
    ) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c / 1_000


def random_route(city_list: List[City]) -> Route:
    route: Route = Route(random.sample(city_list, len(city_list)))
    return route


def random_population(city_list: List[City], population_size: int) -> List[Route]:
    return [random_route(city_list) for _ in range(population_size)]


class RankedRoute(NamedTuple):
    index: int
    route: Route


def rank_routes(population: List[Route]) -> List[RankedRoute]:
    index_paired: List[RankedRoute] = [RankedRoute(i, route) for i, route in enumerate(population)]
    index_paired_sorted: List[RankedRoute] = sorted(index_paired, key=lambda x: x.route.route_fitness, reverse=True)
    return index_paired_sorted


def selection(population_ranked: List[RankedRoute], elite_size: int) -> List[int]:
    selection_result: List[int] = []
    cumsum: np.ndarray = np.array([i.route.route_fitness for i in population_ranked]).cumsum()
    total_sum: float = np.array([i.route.route_fitness for i in population_ranked]).sum()
    cum_percentage: np.ndarray = 100 * cumsum / total_sum

    for i in range(elite_size):
        selection_result.append(population_ranked[i].index)

    for _ in range(len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i, ranked_route in enumerate(population_ranked):
            if cum_percentage[i] >= pick:
                selection_result.append(ranked_route.index)
                break

    return selection_result


def mating_pool(population: List[Route], selection_result: List[int]) -> List[Route]:
    return [population[index] for index in selection_result]


def breed(p1: Route, p2: Route) -> Route:
    child_p1: List[City] = []

    gene_a = int(random.random() * len(p1))
    gene_b = int(random.random() * len(p2))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(p1[i])

    child_p2: List[City] = [item for item in p2.route if item not in child_p1]
    child: List[City] = child_p1 + child_p2
    return Route(child)


def breed_population(population_mating_pool: List[Route], elite_size: int) -> List[Route]:
    children: List[Route] = []
    length = len(population_mating_pool) - elite_size

    pool = random.sample(population_mating_pool, len(population_mating_pool))

    for i in range(elite_size):
        children.append(population_mating_pool[i])

    for i in range(length):
        p1 = pool[i]
        p2 = random.choice(pool)
        child = breed(p1, p2)
        children.append(child)
    return children


def mutate(individual: Route, mutation_rate: float):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def mutate_population(population: List[Route], mutation_rate: float) -> List[Route]:
    return [mutate(i, mutation_rate) for i in population]


def next_generation(current_gen: List[Route], elite_size: int, mutation_rate: float) -> List[Route]:
    population_ranked = rank_routes(current_gen)
    selection_result = selection(population_ranked, elite_size)
    pool = mating_pool(current_gen, selection_result)
    children = breed_population(pool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen


@dataclass(frozen=True)
class History(object):
    fitnesses: List[float]
    distances: List[float]


def genetic_algorithm(city_list: List[City], config: Config) -> Tuple[Route, History]:
    population = random_population(city_list, config.population_size)
    initial_route = rank_routes(population)[0].route
    initial_route_distance = initial_route.route_distance
    initial_route_fitness = initial_route.route_fitness
    st.info(f"Initial Distance: {initial_route_distance:.4f} Km")
    st.info(f"Initial Fitness: {initial_route_fitness}")

    progress = st.progress(0)
    fitnesses: List[float] = []
    distances: List[float] = []

    for i in range(config.num_generations):
        progress.progress((i + 1) / config.num_generations)

        best_route_so_far = rank_routes(population)[0].route
        fitnesses.append(best_route_so_far.route_fitness)
        distances.append(best_route_so_far.route_distance)

        population = next_generation(population, config.elite_size, config.mutation_rate)

    best_route = rank_routes(population)[0].route
    best_route_distance = best_route.route_distance
    best_route_fitness = best_route.route_fitness
    st.success(f"Final Distance: {best_route_distance:.4f} Km")
    st.success(f"Final Fitness: {best_route_fitness}")
    return best_route, History(fitnesses, distances)


M: Dict[str, str] = {
    "name": "Name",
    "country": "Country",
    "timezone": "Timezone",
    "coordinates.latitude": "Latitude",
    "coordinates.longitude": "Longitude"
}


def cities_df(city_list: List[City]) -> pd.DataFrame:
    df_normalized = pd.json_normalize(list(map(asdict, city_list))).set_index("index")
    df_normalized.rename(columns=M, inplace=True)
    return df_normalized


def random_city(i: int, country: str) -> City:
    lat, lon, city, country, tz = fake.local_latlng(country)
    return City(i, Coordinates(float(lat), float(lon)), city, country, tz)


def random_cities(config: Config) -> Optional[List[City]]:
    cities: List[City] = []
    trials: int = 0
    while len(cities) < config.num_cities:
        trials += 1
        city = random_city(len(cities), config.country)
        if city not in cities:
            cities.append(city)
        if trials > config.max_trials:
            st.error(f"Couldn't find unique {config.num_cities} cities in {config.country}")
            return None
    return cities


def construct_pydeck_layer(route: Route) -> pdk.Deck:
    def hex_to_rgb(h: str) -> Tuple[int, ...]:
        h = h.lstrip("#")
        return tuple(int(h[j: j + 2], 16) for j in (0, 2, 4))

    records: List[Dict] = []
    for i, start in enumerate(route.route):
        end = route.route[i + 1] if i + 1 < len(route) else route.route[0]
        record = {
            "name": f"From {i} To {(i + 1) % len(route)}",
            "color": hex_to_rgb(fake.color(hue='red', color_format='hex', luminosity="light")),
            "path": [[start.coordinates.longitude, start.coordinates.latitude],
                     [end.coordinates.longitude, end.coordinates.latitude]]
        }
        records.append(record)
    df = pd.DataFrame.from_records(records)
    view_state = pdk.data_utils.compute_view([[i.coordinates.longitude, i.coordinates.latitude] for i in route.route])

    layer = pdk.Layer(
        type="PathLayer",
        data=df,
        pickable=True,
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=5,
    )

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
    return r


def main():
    st.title("Genetic Algorithm for TSP")
    st.markdown(
        """
        Use a genetic algorithm to find an optimal solution for the traveling salesman problem.
        
        What we are optimizing for is the total
        [haversine distance](https://www.movable-type.co.uk/scripts/latlong.html) traveled by the salesman.
        
        Play with the parameters on the sidebar to see their effect on the optimization process and the result.
        """
    )

    main_run_btn = st.button("Run", key="main_run")

    default_config = Config()

    num_cities = st.sidebar.number_input("Number of Cities", 2, 100, default_config.num_cities, 1)
    population_size = st.sidebar.number_input("Population Size", 5, 10_000, default_config.population_size, 5)
    elite_size = st.sidebar.number_input("Elite Size", 1, population_size, default_config.elite_size, 1)
    mutation_rate = st.sidebar.number_input("Mutation Rate", 0.0, 1.0, default_config.mutation_rate, 0.01)
    generations = st.sidebar.number_input("Number of Generations", 1, 20_000, default_config.num_generations, 50)

    land_coord = faker.providers.geo.Provider.land_coords

    def include_country_code(code: str) -> bool:
        return len([i for i in land_coord if i[3] == code]) >= num_cities

    country_codes: List[str] = faker.providers.address.Provider.alpha_2_country_codes
    valid_country_codes: List[str] = [i for i in country_codes if include_country_code(i)]
    country = st.sidebar.selectbox("Country", valid_country_codes, valid_country_codes.index(default_config.country))

    side_run_btn = st.sidebar.button("Run", key="side_run")

    if main_run_btn or side_run_btn:
        config = Config(num_cities=num_cities, population_size=population_size, elite_size=elite_size,
                        mutation_rate=mutation_rate, num_generations=generations, country=country)
        city_list: Optional[List[City]] = random_cities(config)
        if city_list is not None:
            st.header("Cities")
            st.table(cities_df(city_list))
            best_route, history = genetic_algorithm(city_list, config)
            st.header("History")
            st.line_chart({"Distance": history.distances})
            st.line_chart({"Fitness": history.fitnesses})

            st.header("Best Solution")
            total_route = best_route.route + [best_route[0]]
            st.table(cities_df(total_route))
            st.pydeck_chart(construct_pydeck_layer(best_route))


if __name__ == '__main__':
    main()
