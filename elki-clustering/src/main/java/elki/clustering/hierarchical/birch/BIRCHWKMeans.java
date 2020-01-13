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

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.hierarchical.birch.CFTree.LeafIterator;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.KMeansModel;
import elki.data.model.MeanModel;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDUtil;
import elki.database.ids.ModifiableDBIDs;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.result.Metadata;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.parameterization.Parameterization;

/**
 * BIRCH-based clustering algorithm that simply treats the leafs of the CFTree
 * as clusters.
 * <p>
 * References:
 * <p>
 * T. Zhang, R. Ramakrishnan, M. Livny<br>
 * BIRCH: An Efficient Data Clustering Method for Very Large Databases
 * Proc. 1996 ACM SIGMOD International Conference on Management of Data
 * <p>
 * T. Zhang, R. Ramakrishnan, M. Livny<br>
 * BIRCH: A New Data Clustering Algorithm and Its Applications
 * Data. Min. Knowl. Discovery
 *
 * @author Erich Schubert
 * @since 0.7.5
 *
 * @depend - - - CFTree
 */
@Reference(authors = "T. Zhang, R. Ramakrishnan, M. Livny", //
    title = "BIRCH: An Efficient Data Clustering Method for Very Large Databases", //
    booktitle = "Proc. 1996 ACM SIGMOD International Conference on Management of Data", //
    url = "https://doi.org/10.1145/233269.233324", //
    bibkey = "DBLP:conf/sigmod/ZhangRL96")
@Reference(authors = "T. Zhang, R. Ramakrishnan, M. Livny", //
    title = "BIRCH: A New Data Clustering Algorithm and Its Applications", //
    booktitle = "Data Min. Knowl. Discovery", //
    url = "https://doi.org/10.1023/A:1009783824328", //
    bibkey = "DBLP:journals/datamine/ZhangRL97")
public class BIRCHWKMeans<M extends MeanModel> implements ClusteringAlgorithm<Clustering<MeanModel>> {
  /**
   * CFTree factory.
   */
  CFTree.Factory cffactory;
  
  /**
   * Used weighted k means algorithm.
   */
  LloydWKMeans wkmeans;
  
  /**
   * Constructor.
   *
   * @param cffactory CFTree Factory
   * @param wkmeans LloydWKMeans
   */
  public BIRCHWKMeans(CFTree.Factory cffactory, LloydWKMeans wkmeans) {
    super();
    this.cffactory = cffactory;
    this.wkmeans = wkmeans;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
  }

  /**
   * Run the clustering algorithm.
   *
   * @param relation Input data
   * @return Clustering
   */
  public Clustering<KMeansModel> run(Relation<NumberVector> relation) {
    final int dim = RelationUtil.dimensionality(relation);
    CFTree tree = cffactory.newTree(relation.getDBIDs(), relation);
    // The CFTree does not store points. We have to reassign them (and the
    // quality is better than if we used the initial assignment, because centers
    // move in particular in the beginning, so we always had many outliers.
    ClusteringFeature[] leaves = new ClusteringFeature[tree.leaves];
    double[][] cMeans = new double[tree.leaves][dim];
    double[] cWeights = new double[tree.leaves];
    double[][] cLinSum = new double[tree.leaves][dim];
    double[] cSumSqu = new double[tree.leaves];
    
    int z= 0;
    for(LeafIterator iter = tree.leafIterator();iter.valid();iter.advance()) {
      leaves[z] = iter.get();
      for(int i = 0; i < dim; i++) {
        cMeans[z][i] = leaves[z].centroid(i);
        cLinSum[z][i] = leaves[z].linearSum(i);
      }
      cWeights[z] = (double)leaves[z].n;
      cSumSqu[z] = leaves[z].sumOfSumOfSquares();
      z++;
    }
    List<WCluster<KMeansModel>> clusters = wkmeans.run(cMeans, cWeights, cLinSum, cSumSqu);
    
    Map<ClusteringFeature,WCluster<KMeansModel>> clmap = new HashMap<ClusteringFeature,WCluster<KMeansModel>>();
    Iterator<WCluster<KMeansModel>> cIter = clusters.iterator();
    while(cIter.hasNext()){
      WCluster<KMeansModel> cluster = cIter.next();
      for (int i = 0; i < cluster.size(); i++) {
        clmap.put(leaves[cluster.getIDs().get(i)], cluster);
      }
    }

    Map<WCluster<KMeansModel>,ModifiableDBIDs> idmap = new HashMap<WCluster<KMeansModel>,ModifiableDBIDs>();
    for(DBIDIter iter = relation.iterDBIDs(); iter.valid(); iter.advance()) {
      ClusteringFeature cf = tree.findLeaf(relation.get(iter));
      WCluster<KMeansModel> cl = clmap.get(cf); 
      ModifiableDBIDs ids = idmap.get(cl);
      if(ids == null) {
        idmap.put(clmap.get(cf), ids = DBIDUtil.newArray());
      }
      ids.add(iter);
    }     
    Clustering<KMeansModel> result = new Clustering<>();
    for(Map.Entry<WCluster<KMeansModel>, ModifiableDBIDs> ent : idmap.entrySet()) {
      result.addToplevelCluster(new Cluster<KMeansModel>(ent.getValue(), ent.getKey().getModel()));
    }
    Metadata.of(result).setLongName("BIRCH weighted k Means Clustering");
    return result;
  }

  /**
   * Parameterization class.
   *
   * @author Erich Schubert
   */
  public static class Par<M extends MeanModel> implements Parameterizer {
    /**
     * CFTree factory.
     */
    CFTree.Factory cffactory;
    
    LloydWKMeans wkmeans;

    @Override
    public void configure(Parameterization config) {
      cffactory = config.tryInstantiate(CFTree.Factory.class);
      wkmeans = config.tryInstantiate(LloydWKMeans.class);
    }

    @Override
    public BIRCHWKMeans<M> make() {
      return new BIRCHWKMeans<M>(cffactory, wkmeans);
    }
  }
}
