/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2020
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
package elki.clustering.hierarchical.birch;

import java.util.*;

import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

/**
 * K-Means++ initialization for weighted k-means.
 * <p>
 * Reference:
 * <p>
 * D. Arthur, S. Vassilvitskii<br>
 * k-means++: the advantages of careful seeding<br>
 * Proc. 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)
 *
 * @author Andreas Lang
 * 
 * @since 0.7.5
 *
 */
public class WKMeansPlusPlus {
  
  /**
   * Random generator
   */
  protected Random random;
  
  /**
   * Weights
   */
  protected double[] weights;
  
  /**
   * Constructor.
   *
   * @param rnd Random generator.
   */
  public WKMeansPlusPlus(RandomFactory rnd) {
    this.random = rnd.getSingleThreadedRandom();
  }
  
  /**
   * Run k-means++ initialization for double.
   *
   * @param k K
   * @param x Input vectors.
   * @return Vectors
   */
  public double[][] run(final double[][] x, int k){
    List<Integer> means = new ArrayList<>(k);
    int first = random.nextInt(x.length);
    means.add(first);
    chooseRemaining(k, means, initialWeights(first, x), x);
    return build(means,x);
  }
  
  /**
   * Build means.
   * 
   * @param meanIds Ids of chosen Mean.
   * @param x Data
   * @return Means
   */
  private double[][] build(List<Integer> meanIds, double[][] x) {
    Iterator<Integer> it = meanIds.listIterator();
    double[][] means = new double[meanIds.size()][x[0].length];
    int i = 0;
    while(it.hasNext()) {
      int id = it.next();
      means[i] = Arrays.copyOf(x[id], x[id].length);;
      i++;
    }
    return means;
  }

  /**
     * Choose remaining means, weighted by distance.
     *
     * @param k Number of means to choose
     * @param means Means storage
     * @param weightsum Sum of weights
   */
  private void chooseRemaining(int k, List<Integer> means, double weightsum, double[][] x) {
    while(true) {
      if(weightsum > Double.MAX_VALUE) {
        throw new IllegalStateException("Could not choose a reasonable mean - too many data points, too large distance sum?");
      }
      if(weightsum < Double.MIN_NORMAL) {
        //LOG.warning("Could not choose a reasonable mean - to few unique data points?");
      }
      double r = nextDouble(weightsum);
      int i = 0;
      while (i < x.length) {
        if((r -= weights[i]) <= 0) {
          break;
        }
        i++;
      }
      if (i >= x.length) { // Rare case, but happens due to floating math
        weightsum -= r; // Decrease
        continue; // Retry
      }
      // Add new mean:
      final int newmean = i;
      means.add(newmean);
      if(means.size() >= k) {
        break;
      }
      // Update weights:
      weights[i] = 0.;
      weightsum = updateWeights(newmean,x);
    }
  }
  
  private double nextDouble(double weightsum) {
    double r = random.nextDouble() * weightsum;
    while(r <= 0 && weightsum > Double.MIN_NORMAL) {
      r = random.nextDouble() * weightsum; // Try harder to not choose 0.
    }
    return r;
  }

  /**
   * Initialize the weight list.
   *  
   * @param first Id of first mean.
   * @param x Input data.
   * @return Sum of weights
   */
  private double initialWeights(int first, double[][] x) {
    double weightsum = 0.;
    weights = new double[x.length];
    for(int i = 0; i < x.length; i++) {
      weights[i] = distance(x[i], x[first]);
      weightsum += weights[i];
    }
    return weightsum;
  }
  
  /**
   * Calculates distance between two vectors.
   * 
   * @param ds First Vector
   * @param ds2 Second Vector
   * @return Vector
   */
  private double distance(double[] ds, double[] ds2) {
    double v = 0;
    for(int i = 0; i < ds.length; i++) {
      double d = ds[i] - ds2[i];
      v += d * d;
    }
    return v;
  }

  /**
   * Update the weight list.
   *
   * @param latest Added ID
   * @return Weight sum
   */
  private double updateWeights(int latest, double[][]x) {
    double weightsum = 0.;
    for(int i = 0; i < x.length; i++) {
      double weight = weights[i];
      if(weight <= 0.) {
        continue; // Duplicate, or already chosen.
      }
      double newweight = distance(x[latest], x[i]);
      if(newweight < weight) {
        weights[i] = newweight;
        weight = newweight;
      }
      weightsum += weight;
    }
    return weightsum;
  }
  
  /**
   * Parameterization class.
   * 
   */
  public static class Par implements Parameterizer {
    /**
     * Random generator
     */
    protected RandomFactory rnd;
    
    /**
     * Parameter to specify the random generator seed.
     */
    OptionID SEED_ID = new OptionID("kmeans.seed", "The random number generator seed.");

    @Override
    public void configure(Parameterization config) {
      new RandomParameter(SEED_ID).grab(config, x -> rnd = x);
    }
    
    @Override
    public WKMeansPlusPlus make(){
      return new WKMeansPlusPlus(rnd);
    }
  }
}
