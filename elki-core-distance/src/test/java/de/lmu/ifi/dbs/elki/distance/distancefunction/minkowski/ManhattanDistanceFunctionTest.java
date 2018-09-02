/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2018
 * ELKI Development Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package de.lmu.ifi.dbs.elki.distance.distancefunction.minkowski;

import org.junit.Test;

import de.lmu.ifi.dbs.elki.distance.distancefunction.AbstractSpatialPrimitiveDistanceFunctionTest;
import de.lmu.ifi.dbs.elki.utilities.ELKIBuilder;

/**
 * Unit test for Manhattan distance.
 *
 * @author Erich Schubert
 */
public class ManhattanDistanceFunctionTest extends AbstractSpatialPrimitiveDistanceFunctionTest {
  @Test
  public void testSpatialConsistency() {
    // Also test the builder - we could have just used .STATIC
    ManhattanDistanceFunction dis = new ELKIBuilder<>(ManhattanDistanceFunction.class).build();
    spatialConsistency(dis);
    nonnegativeSpatialConsistency(dis);
  }
}