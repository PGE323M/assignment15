#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import numpy as np
import scipy.sparse

from assignment15 import TwoDimReservoir

class TestSolution(unittest.TestCase):

    def setUp(self):


        self.inputs = {
            'conversion factor': 6.33e-3,
            'fluid': {
                'water': {
                    'compressibility': 1e-6, #psi^{-1}
                    'viscosity': 1, #cp
                    'formation volume factor': 1,
                    },
            },
            'reservoir': {
                'permeability': 50, #md
                'porosity': 0.2,
                'length': 10000, #ft
                'height': 200000, #ft
                'depth': 1 #ft
            },
            'initial conditions': {
                'pressure': 1000 #psi
            },
            'boundary conditions': {
                'left': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'right': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #psi
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
            },
            'numerical': {
                'solver': 'implicit',
                'number of grids': {'x': 4, 'y': 1},
                'time step': 1, #day
                'number of time steps' : 3
            },
            'plots': {
                'frequency': 1
            }
        }

        return

    def test_compute_transmissibility(self):

        problem = TwoDimReservoir(self.inputs)

        np.testing.assert_allclose(problem.compute_transmissibility(0,0), 4000.0)

        return

    def test_compute_accumulation(self):


        problem = TwoDimReservoir(self.inputs)

        np.testing.assert_allclose(problem.compute_accumulation(0), 100.0)

        return

    def test_is_transmissiblity_matrix_sparse(self):


        problem = TwoDimReservoir(self.inputs)

        assert scipy.sparse.issparse(problem.T)

        return

    def test_implicit_solve_one_step(self):

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]),
                                   atol=0.5)
        return

    def test_explicit_solve_one_step(self):

        self.inputs['numerical']['solver'] = 'explicit'

        explicit = TwoDimReservoir(self.inputs)

        explicit.solve_one_step()

        np.testing.assert_allclose(explicit.get_solution(),
                               np.array([ 1506., 1000.,  1000.,  1000.004]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_implicit(self):

        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.0}}

        mixed_implicit = TwoDimReservoir(self.inputs)

        mixed_implicit.solve_one_step()

        np.testing.assert_allclose(mixed_implicit.get_solution(),
                               np.array([1295.1463, 1051.1036, 1008.8921, 1001.7998]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_explicit(self):


        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 1.0}}

        mixed_explicit = TwoDimReservoir(self.inputs)

        mixed_explicit.solve_one_step()

        np.testing.assert_allclose(mixed_explicit.get_solution(),
                               np.array([ 1506., 1000.,  1000.,  1000.004]),
                               atol=0.5)
        return

    def test_mixed_method_solve_one_step_crank_nicolson(self):


        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.5}}

        mixed = TwoDimReservoir(self.inputs)

        mixed.solve_one_step()

        np.testing.assert_allclose(mixed.get_solution(),
                                   np.array([ 1370.4,  1037.8 ,  1003.8,  1000.4]),
                                   atol=0.5)
        return

    def test_implicit_solve(self):

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1582.9, 1184.8, 1051.5, 1015.9]),
                                   atol=0.5)
        return

    def test_implicit_solve_reverse_boundary_conditions(self):


        self.inputs['boundary conditions'] = {
                'right': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'left': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
            }

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1015.9, 1051.5, 1184.8, 1582.9]),
                                   atol=0.5)
        return

    def test_explicit_solve(self):

        self.inputs['numerical']['solver'] = 'explicit'

        explicit = TwoDimReservoir(self.inputs)

        explicit.solve()

        np.testing.assert_allclose(explicit.get_solution(),
                               np.array([1689.8, 1222.3, 1032.4, 1000.0]),
                               atol=0.5)
        return

    def test_mixed_method_solve_crank_nicolson(self):

        self.inputs['numerical']['solver'] = {'mixed method': {'theta': 0.5}}

        mixed = TwoDimReservoir(self.inputs)

        mixed.solve()

        np.testing.assert_allclose(mixed.get_solution(),
                                   np.array([ 1642.0,  1196.5,  1043.8,  1009.1]),
                                   atol=0.5)
        return

    def test_implicit_heterogeneous_permeability_solve_one_step(self):


        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1085.3,  1005.8,  1001.3,  1000.1]),
                                   atol=0.5)
        return

    def test_implicit_heterogeneous_permeability_and_grid_size_solve_one_step(self):


        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]
        self.inputs['numerical']['delta x'] = [2000., 3000., 1500., 3500]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1123.0,  1008.5,  1003.1,  1000.2]),
                                   atol=0.5)
        return


    def test_implicit_heterogeneous_permeability_and_grid_size_solve(self):


        self.inputs['reservoir']['permeability'] = [10., 100., 50., 20]
        self.inputs['numerical']['delta x'] = [2000., 3000., 1500., 3500]

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([1295.6,  1039.1,  1019.9,  1002.5]),
                                   atol=0.5)
        return

    def test_two_dim_solve_one_step(self):


        self.inputs['numerical']['number of grids'] = {'x': 3, 'y': 3}
        self.inputs['reservoir']['height'] = 10000.
        self.inputs['reservoir']['depth'] = 20.
        self.inputs['boundary conditions'] = {
                'right': {
                    'type': 'prescribed pressure',
                    'value': 2000 #psi
                },
                'left': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                },
                'top': {
                    'type': 'prescribed flux',
                    'value': 0 #psi
                },
                'bottom': {
                    'type': 'prescribed flux',
                    'value': 0 #ft^3/day
                }
        }

        implicit = TwoDimReservoir(self.inputs)
        implicit.solve_one_step()
        np.testing.assert_allclose(implicit.get_solution(),
                                   np.array([ 1002.8,  1022.6,  1201.8,  1002.8, 1022.6,
                                              1201.8,  1002.8,  1022.6, 1201.8]),
                                   atol=0.5)
        return





class TestTwoDimensionalContract(unittest.TestCase):
    def inputs(self):
        case = TestSolution()
        case.setUp()
        return case.inputs

    def box(self, nx=2, ny=2):
        inputs = self.inputs()
        inputs['numerical']['number of grids'] = {'x': nx, 'y': ny}
        inputs['numerical']['time step'] = 0.05
        inputs['reservoir'].update(length=4., height=6., depth=2., permeability=3.)
        for bc in inputs['boundary conditions'].values():
            bc.update(type='prescribed flux', value=0)
        return inputs

    def test_inherits_time_integrator(self):
        from assignment13 import OneDimReservoir
        self.assertTrue(issubclass(TwoDimReservoir, OneDimReservoir))
        self.assertIs(TwoDimReservoir.solve_one_step, OneDimReservoir.solve_one_step)

    def test_two_by_two_matrix_no_row_wrap(self):
        p = TwoDimReservoir(self.box())
        # dx=2, dy=3, depth=2, k=3: horizontal=9, vertical=4.
        expected = np.array([[13, -9, -4, 0], [-9, 13, 0, -4],
                             [-4, 0, 13, -9], [0, -4, -9, 13]]) * p.conversion_factor
        np.testing.assert_allclose(p.T.toarray(), expected)
        np.testing.assert_allclose(p.T.toarray(), p.T.toarray().T)
        np.testing.assert_allclose(p.T.sum(axis=1), 0, atol=1e-15)
        self.assertTrue(scipy.sparse.issparse(p.B))
        self.assertEqual(p.T.format, 'csr')
        self.assertEqual(p.B.format, 'csr')

    def test_vertical_pressure_boundary_and_corner(self):
        inputs = self.box()
        inputs['boundary conditions']['top'] = {'type': 'prescribed pressure', 'value': 1200}
        inputs['boundary conditions']['left'] = {'type': 'prescribed pressure', 'value': 900}
        p = TwoDimReservoir(inputs)
        f = p.conversion_factor
        np.testing.assert_allclose(p.Q, [18*900*f, 0, (18*900+8*1200)*f, 8*1200*f])
        np.testing.assert_allclose(p.T.diagonal(), np.array([31, 13, 39, 21])*f)

    def test_heterogeneous_half_block_resistances_and_storage(self):
        inputs = self.box()
        inputs['numerical']['delta x'] = [1., 3.]
        inputs['numerical']['delta y'] = [2., 4.]
        inputs['reservoir'].update(permeability=[2., 5., 3., 7.],
                                   depth=[1., 2., 3., 4.], porosity=[.1, .2, .3, .4])
        p = TwoDimReservoir(inputs)
        # Face conductances from the two half-block resistances.
        np.testing.assert_allclose(p.compute_transmissibility(0, 1), 1/(1/8+3/40))
        np.testing.assert_allclose(p.compute_transmissibility(0, 2), 1/(2/4+4/18))
        np.testing.assert_allclose(p.compute_transmissibility(2, 0), p.compute_transmissibility(0, 2))
        np.testing.assert_allclose(p.B.diagonal(), np.array([.2, 2.4, 3.6, 19.2])*1e-6)

    def test_impermeable_cells_remain_finite(self):
        inputs = self.box()
        inputs['reservoir']['permeability'] = [0., 0., 2., 3.]
        p = TwoDimReservoir(inputs)
        for i, j in [(0, 1), (0, 2), (1, 3)]:
            self.assertEqual(p.compute_transmissibility(i, j), 0)
        self.assertTrue(np.isfinite(p.T.data).all())

    def test_one_column_uses_vertical_connections(self):
        p = TwoDimReservoir(self.box(nx=1, ny=2))
        # dx=4, dy=3: the single internal connection is vertical, not horizontal.
        np.testing.assert_allclose(p.compute_transmissibility(0, 1), 8.)

    def test_closed_uniform_reservoir_stays_uniform(self):
        p = TwoDimReservoir(self.box(nx=3, ny=2))
        p.solve_one_step()
        np.testing.assert_allclose(p.get_solution(), 1000, atol=1e-7)

    def test_one_row_matches_assignment13_for_all_time_methods(self):
        import copy
        from assignment13 import OneDimReservoir
        for solver in ['explicit', 'implicit', {'mixed method': {'theta': .5}}]:
            inputs = self.inputs()
            inputs['numerical']['solver'] = solver
            inputs['reservoir']['permeability'] = [10., 100., 50., 20.]
            inputs['numerical']['delta x'] = [2000., 3000., 1500., 3500.]
            base_inputs = copy.deepcopy(inputs)
            base_inputs['reservoir']['cross sectional area'] = 200000.
            base = OneDimReservoir(base_inputs)
            p = TwoDimReservoir(inputs)
            base.solve()
            p.solve()
            np.testing.assert_allclose(p.get_solution(), base.get_solution(), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
