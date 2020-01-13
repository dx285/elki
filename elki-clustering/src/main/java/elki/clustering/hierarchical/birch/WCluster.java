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

import java.util.List;

import elki.data.model.Model;
import elki.result.textwriter.TextWriteable;
import elki.result.textwriter.TextWriterStream;

/*
 * TODO
 *
 * @author Andreas Lang
 * 
 * @since 0.7.5
 *
 */
public class WCluster <M extends Model> implements TextWriteable{

  
  private List<Integer> ids;
  private String name;
  
  /**
   * Cluster model.
   */
  private M model = null;
  
  /**
   * Full constructor
   * 
   * @param name Cluster name. May be null.
   * @param ids Object Group
   * @param noise Noise flag
   * @param model Model. May be null.
   */
  public WCluster(String name, List<Integer> ids, M model) {
    this.name = name;
    this.ids = ids;
    this.model = model;
  }
  
  /**
   * Delegate to database object group.
   * 
   * @return Cluster size retrieved from object group.
   */
  public int size() {
    return ids.size();
  }

  /**
   * Return either the assigned name or the suggested label
   * 
   * @return a name for the cluster
   */
  public String getNameAutomatic() {
    if(name != null) {
      return name;
    }
    else {
      return "Cluster";
    }
  }
  
  /**
   * Get Cluster name. May be null.
   * 
   * @return cluster name, or null
   */
  public String getName() {
    return name;
  }

  /**
   * Set Cluster name
   * 
   * @param name new cluster name
   */
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Access group object
   * 
   * @return database object group
   */
  public List<Integer> getIDs() {
    return ids;
  }

  /**
   * Access group object
   * 
   * @param g set database object group
   */
  public void setIDs(List<Integer> g) {
    ids = g;
  }

  /**
   * Access model object
   * 
   * @return Cluster model
   */
  public M getModel() {
    return model;
  }

  /**
   * Access model object
   * 
   * @param model New cluster model
   */
  public void setModel(M model) {
    this.model = model;
  }
  
  /**
   * Write to a textual representation. Writing the actual group data will be
   * handled by the caller, this is only meant to write the meta information.
   * 
   * @param out output writer stream
   * @param label Label to prefix
   */
  @Override
  public void writeToText(TextWriterStream out, String label) {
    String name = getNameAutomatic();
    if(name != null) {
      out.commentPrintLn("Cluster name: " + name);
    }
    out.commentPrintLn("Cluster size: " + ids.size());
    // also print model, if any and printable
    if(getModel() != null && (getModel() instanceof TextWriteable)) {
      ((TextWriteable) getModel()).writeToText(out, label);
    }
    
  }
  
}
