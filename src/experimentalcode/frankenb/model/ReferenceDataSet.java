package experimentalcode.frankenb.model;

import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.ids.DBIDUtil;
import de.lmu.ifi.dbs.elki.database.ids.HashSetModifiableDBIDs;
import experimentalcode.frankenb.model.ifaces.IDataSet;

/**
 * This class behaves like the DataSet class but just contains a list of item
 * ids and not the associated vectors with it. This is especially usefull when the
 * data set is e.g. splitted.
 * 
 * @author Florian Frankenberger
 */
public class ReferenceDataSet implements IDataSet {

  private final IDataSet originalDataSet;
  private HashSetModifiableDBIDs items = DBIDUtil.newHashSet();
  
  public ReferenceDataSet(IDataSet originalDataSet) {
    if (originalDataSet == null) throw new RuntimeException("original data set can't be null!");
    this.originalDataSet = originalDataSet;
  }
  
  @Override
  public NumberVector<?, ?> get(DBID id) {
    if (!this.items.contains(id)) return null;
    return this.originalDataSet.get(id);
  }

  @Override
  public int getDimensionality() {
    return this.originalDataSet.getDimensionality();
  }

  @Override
  public int getSize() {
    return this.items.size();
  }
  
  public void add(DBID id) {
    this.items.add(id);
  }

  @Override
  public IDataSet getOriginal() {
    return this.originalDataSet;
  }

  @Override
  public Iterable<DBID> getIDs() {
    return items;
  }
  
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    boolean first = true;
    sb.append("{");
    for (DBID id : this.getIDs()) {
      if (first) {
        first = false;
      } else
        sb.append(", ");
      sb.append("(").append(this.get(id)).append(")").append(" [").append(id).append("]");
    }
    sb.append("}");
    return sb.toString();
  }

}
