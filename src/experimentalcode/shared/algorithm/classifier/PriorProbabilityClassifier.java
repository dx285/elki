package experimentalcode.shared.algorithm.classifier;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2014
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.ArrayList;

import de.lmu.ifi.dbs.elki.algorithm.AbstractAlgorithm;
import de.lmu.ifi.dbs.elki.data.ClassLabel;
import de.lmu.ifi.dbs.elki.data.type.TypeInformation;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.ids.DBIDIter;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.result.Result;
import de.lmu.ifi.dbs.elki.utilities.documentation.Description;
import de.lmu.ifi.dbs.elki.utilities.documentation.Title;

/**
 * Classifier to classify instances based on the prior probability of classes in
 * the database.
 * 
 * @author Arthur Zimek
 *
 * @param <L> the type of the ClassLabel the Classifier is assigning
 */
@Title("Prior Probability Classifier")
@Description("Classifier to predict simply prior probabilities for all classes as defined by their relative abundance in a given database.")
public class PriorProbabilityClassifier<L extends ClassLabel> extends AbstractAlgorithm<Result> implements Classifier<Object, L> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(PriorProbabilityClassifier.class);

  /**
   * Holds the prior probabilities.
   */
  protected double[] distribution;

  /**
   * Index of the most abundant class.
   */
  protected L prediction;

  /**
   * Class labels seen.
   */
  protected ArrayList<L> labels;

  /**
   * Provides a classifier always predicting the prior probabilities.
   */
  public PriorProbabilityClassifier() {
    super();
  }

  /**
   * Learns the prior probability for all classes.
   */
  @Override
  public void buildClassifier(Database database, Relation<L> labelrep) {
    TObjectIntMap<L> count = new TObjectIntHashMap<>();
    for(DBIDIter iter = labelrep.iterDBIDs(); iter.valid(); iter.advance()) {
      count.adjustOrPutValue(labelrep.get(iter), 1, 1);
    }
    int max = Integer.MIN_VALUE;
    double size = labelrep.size();

    distribution = new double[count.size()];
    labels = new ArrayList<>(count.size());
    TObjectIntIterator<L> iter = count.iterator();
    for(int i = 0; iter.hasNext(); ++i) {
      iter.advance();
      distribution[i] = iter.value() / size;
      labels.add(iter.key());
      if(iter.value() > max) {
        max = iter.value();
        prediction = iter.key();
      }
    }
  }

  public double[] classProbabilities(Object instance, ArrayList<L> labels) {
    // FIXME: labels may be sorted differently!
    return distribution;
  }
  
  @Override
  public L classify(Object instance) {
    return prediction;
  }

  @Override
  public String model() {
    StringBuilder output = new StringBuilder();
    for(int i = 0; i < distribution.length; i++) {
      output.append(labels.get(i));
      output.append(" : ");
      output.append(distribution[i]);
      output.append('\n');
    }
    return output.toString();
  }

  @Override
  public Result run(Database database) {
    // FIXME Implement sensible default behavior.
    return null;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(TypeUtil.ANY);
  }

  @Override
  protected Logging getLogger() {
    return LOG;
  }
}