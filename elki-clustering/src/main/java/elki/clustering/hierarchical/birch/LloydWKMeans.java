/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2019
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

import static elki.math.linearalgebra.VMath.timesEquals;
import static elki.math.linearalgebra.VMath.plusEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import elki.data.model.KMeansModel;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;


/**
 * Lloyd's k-means algorithm with weighted elements for usage with BIRCH.
 *
 * @author Andreas Lang
 * @since 0.7.5
 *
 */
public class LloydWKMeans {
  
  /**
   * k-means++ initialization 
   */
  WKMeansPlusPlus initialization;
  
  /**
   * Maximum number of iterations.
   */
  int maxIter;
  
  /**
   * Number of cluster centers to initialize.
   */
  int k;
  
  /**
   * Number of dimensions of input data.
   */
  int dim;
  
  /**
   * Cluster means.
   */
  double[][]means;
  
  /**
   * Stored elements per cluster.
   */
  ArrayList<ArrayList<Integer>> clusters;
  
  /**
   * number of elements.
   */
  int count;
  
  /**
   * Means of input data .
   */
  double[][] x;
  
  /**
   * Weights of the elements of input data.
   */
  double[] weights;
  
  /**
   * Linear sum of elements of input data.
   */
  double[][] linsum;
  
  /**
   * Sum of sum of squares of elements of input data.
   */
  double[] sumsqu;
  
  /**
   * A mapping of elements to cluster ids.
   */
  int[] assignment;
  
  /**
   * Sum of squared deviations in each cluster.
   */
  double[] varsum;
  
  /**
   * Constructor.
   *
   * @param k k parameter
   * @param maxIter Maxiter parameter
   */
  public LloydWKMeans(int k, int maxIter, WKMeansPlusPlus initialization) {
    this.initialization = initialization;
    this.maxIter = maxIter > 0 ? maxIter : Integer.MAX_VALUE;
    this.k = k;
    varsum = new double[k];
    clusters = new ArrayList<>(k);
    for(int i = 0; i < k;i++) {
      clusters.add(new ArrayList<>());
    }
    
  }

  /**
   * Run the clustering algorithm.
   *
   * @param cMeans Means of elements.
   * @param cWeigths Weights of elements.
   * @param cLinSum Linear sum of elements.
   * @param cSumSqu Sum of sum of squares of elements.
   * @return Clustering result
   */
  public List<WCluster<KMeansModel>> run(final double[][] cMeans, final double[] cWeights, final double[][] cLinSum, final double[] cSumSqu) {
    count = cWeights.length;
    x = cMeans;
    weights = cWeights;
    linsum = cLinSum;
    sumsqu = cSumSqu;
    assignment = new int[count];
    dim = cMeans[0].length;
    
    means = initialization.run(x, k);
    for( int i = 1; i <= maxIter; i++) {
      int changed = iterate(i);
      if(changed==0) {
        break;
      }
    }
    return buildResult();
    
  }
  
  /**
   * Main loop of Lloyd's kMeans.
   * 
   * @param iteration Iteration number.
   * @return Number of reassigned elements.
   */
  private int iterate(int iteration) {
    means = iteration == 1 ? means : means(clusters, means, linsum, weights);
    return assignToNearestCluster();
  }
  
  /**
   * Calculate means of clusters.
   * 
   * @param clusters elements of clusters.
   * @param means Means of clusters.
   * @param linsum Linear sum of the elements;
   * @param weights Weights of the elements;
   * @return Means of clusters.
   */
  private double[][] means(ArrayList<ArrayList<Integer>> clusters, double[][] means, double[][] linsum, double[]weights) {
    double[][] newMeans = new double[k][];
    for(int i = 0; i < k ;i++) {
      List<Integer> cluster = clusters.get(i);
      if(cluster.isEmpty()) {
        // Keep degenerated means as-is for now.
        // TODO: allow the user to choose the behavior.
        newMeans[i] = means[i];
        continue;
      }
      Iterator<Integer> iter = cluster.iterator();
      int id = iter.next();
      double[] sum = Arrays.copyOf(linsum[id],linsum[id].length);
      double weight = weights[id];
      while(iter.hasNext()) {
        id = iter.next();
        sum = plusEquals(sum, linsum[id]);
        weight += weights[id];
      }
      newMeans[i] = timesEquals(sum, 1.0 / weight);
    }
    return newMeans;
  }

  /**
   * Assign each element to nearest cluster.
   * 
   * @return Number of reassigned elements.
   */
  private int assignToNearestCluster() {
    assert (k == means.length);
    int changed = 0;
    Arrays.fill(varsum, 0.);
    for(List<Integer> cluster : clusters) {
      cluster.clear();
    }
    for(int i = 0; i < count; i++ ) {
      double mindist = distance(x[i], means[0]);
      int minIndex = 0;
      for(int j = 1; j < k; j++) {
        double dist = distance(x[i], means[j]);
        if(dist < mindist) {
          minIndex = j;
          mindist = dist;
        }
      }
    varsum[minIndex] += mindist;
    clusters.get(minIndex).add(i);
    if (assignment[i] != minIndex) {
      changed++;
      assignment[i] = minIndex;
    }
    }
    return changed;
  } 

  /**
   * Calculates the squared euclidean distance between two objects.
   * 
   * @param ds First object.
   * @param ds2 Second object.
   * @return Distance between objects.
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
   * Build result for weighted kMeans clustering.
   * 
   * @return Clustering result.
   */
  private List<WCluster<KMeansModel>> buildResult(){ 
  List<WCluster<KMeansModel>> result = new ArrayList<>(k);
  for(int i = 0; i < k; i++) {
    double[] mean = means[i];
    double varsum = calculateVariance(i);

    result.add(new WCluster<>(null, clusters.get(i), new KMeansModel(mean, varsum)));
  }
  return result;
  }
  
  /**
   * Calculate variance of Clusters based on clustering features.
   * 
   * @param clusterId Id of Cluster.
   * @return Variance
   */
  private double calculateVariance(int clusterId) {
    double[] ls = new double[dim];
    double ss = 0;
    double weight = 0;
    List<Integer> cluster = clusters.get(clusterId);
    for (int id : cluster) {
      plusEquals(ls,linsum[id]);
      ss += sumsqu[id];
      weight += weights[id];
    }
    double lsSum = 0.;
    for(int i = 0; i < dim; i++) {
      lsSum += ls[i] * ls[i];
    }
    double result = ss * weight - lsSum;
    return result/(weight * weight);

  }
  
  public static class Par implements Parameterizer {
    /**
     * k Parameter.
     */
    protected int k;

    /**
     * Maximum number of iterations.
     */
    protected int maxiter;
    
    /**
     * initialization method
     */
    protected WKMeansPlusPlus initialization;
    
    /**
     * Parameter to specify the number of clusters to find, must be an integer
     * greater than 0.
     */
    OptionID K_ID = new OptionID("kmeans.k", "The number of clusters to find.");

    /**
     * Parameter to specify the number of clusters to find, must be an integer
     * greater or equal to 0, where 0 means no limit.
     */
    OptionID MAXITER_ID = new OptionID("kmeans.maxiter", "The maximum number of iterations to do. 0 means no limit.");

    @Override
    public void configure(Parameterization config) {
      getParameterK(config);
      getParameterMaxIter(config);
      initialization = config.tryInstantiate(WKMeansPlusPlus.class);
    }

    /**
     * Get the k parameter.
     *
     * @param config Parameterization
     */
    protected void getParameterK(Parameterization config) {
      new IntParameter(K_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
          .grab(config, x -> k = x);
    }
    
    /**
     * Get the max iterations parameter.
     *
     * @param config Parameterization
     */
    protected void getParameterMaxIter(Parameterization config) {
      new IntParameter(MAXITER_ID, 0)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .grab(config, x -> maxiter = x);
    }

    @Override
    public LloydWKMeans make() {
      return new LloydWKMeans(k, maxiter, initialization);
    }
  }
  
}
