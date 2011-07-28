package de.lmu.ifi.dbs.elki.datasource;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import de.lmu.ifi.dbs.elki.data.LabelList;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.datasource.bundle.MultipleObjectsBundle;
import de.lmu.ifi.dbs.elki.datasource.filter.ObjectFilter;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.utilities.FormatUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Parameterizable;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameterization.Parameterization;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.ObjectListParameter;

/**
 * Joins multiple data sources by their label
 * 
 * @author Erich Schubert
 */
public class LabelJoinDatabaseConnection extends AbstractDatabaseConnection implements Parameterizable {
  /**
   * Logger
   */
  private static final Logging logger = Logging.getLogger(LabelJoinDatabaseConnection.class);

  /**
   * The filters to invoke
   */
  final protected List<DatabaseConnection> sources;

  /**
   * Constructor.
   * 
   * @param filters Filters to use.
   * @param sources Data sources to join.
   */
  public LabelJoinDatabaseConnection(List<ObjectFilter> filters, List<DatabaseConnection> sources) {
    super(filters);
    this.sources = sources;
  }

  @Override
  public MultipleObjectsBundle loadData() {
    List<MultipleObjectsBundle> bundles = new ArrayList<MultipleObjectsBundle>(sources.size());
    for(DatabaseConnection dbc : sources) {
      bundles.add(dbc.loadData());
    }

    MultipleObjectsBundle first = bundles.get(0);
    Map<String, Integer> labelmap = new HashMap<String, Integer>(first.dataLength());
    // Process first bundle
    {
      // Identify a label column
      final int lblcol;
      {
        int lblc = -1;
        for(int i = 0; i < first.metaLength(); i++) {
          if(TypeUtil.GUESSED_LABEL.isAssignableFromType(first.meta(i))) {
            lblc = i;
            break;
          }
        }
        lblcol = lblc; // make static
      }
      for(int i = 0; i < first.dataLength(); i++) {
        Object data = first.data(i, lblcol);
        if(data == null) {
          logger.warning("Object without label encountered.");
          continue;
        }
        if(data instanceof String) {
          Integer old = labelmap.put((String) data, i);
          if(old != null) {
            logger.warning("Duplicate label encountered: " + data + " in rows " + old + " and " + i);
          }
        }
        else if(data instanceof LabelList) {
          for(String lbl : (LabelList) data) {
            Integer old = labelmap.put(lbl, i);
            if(old != null) {
              logger.warning("Duplicate label encountered: " + lbl + " in rows " + old + " and " + i);
            }
          }
        }
        else {
          String lbl = data.toString();
          Integer old = labelmap.put(lbl, i);
          if(old != null) {
            logger.warning("Duplicate label encountered: " + lbl + " in rows " + old + " and " + i);
          }
        }
      }
    }
    // Process additional columns
    for(int c = 1; c < sources.size(); c++) {
      MultipleObjectsBundle cur = bundles.get(c);
      final int lblcol;
      {
        int lblc = -1;
        for(int i = 0; i < cur.metaLength(); i++) {
          if(TypeUtil.GUESSED_LABEL.isAssignableFromType(cur.meta(i))) {
            lblc = i;
            break;
          }
        }
        lblcol = lblc; // make static
      }
      // Destination columns
      List<ArrayList<Object>> dcol = new ArrayList<ArrayList<Object>>(cur.metaLength());
      for(int i = 0; i < cur.metaLength(); i++) {
        // Skip the label columns
        if(i == lblcol) {
          dcol.add(null);
          continue;
        }
        ArrayList<Object> newcol = new ArrayList<Object>(first.dataLength());
        // Pre-fill with nulls.
        for(int j = 0; j < first.dataLength(); j++) {
          newcol.add(null);
        }
        first.appendColumn(cur.meta(i), newcol);
        dcol.add(newcol);
      }
      for(int i = 0; i < cur.dataLength(); i++) {
        Object data = cur.data(i, lblcol);
        if(data == null) {
          logger.warning("Object without label encountered.");
          continue;
        }
        Integer row = null;
        if(data instanceof String) {
          row = labelmap.get(data);
        }
        else if(data instanceof LabelList) {
          for(String lbl : (LabelList) data) {
            row = labelmap.get(lbl);
            if(row != null) {
              break;
            }
          }
        }
        else {
          row = labelmap.get(data.toString());
        }
        if(row == null) {
          logger.warning("Label not found for join: " + data + " in row " + i);
          continue;
        }
        for(int d = 0; d < cur.metaLength(); d++) {
          if(d == lblcol) {
            continue;
          }
          List<Object> col = dcol.get(d);
          assert (col != null);
          col.set(row, cur.data(i, d));
        }
      }
    }
    for(int i = 0; i < first.dataLength(); i++) {
      for(int d = 0; d < first.metaLength(); d++) {
        if(first.data(i, d) == null) {
          StringBuffer buf = new StringBuffer();
          for(int d2 = 0; d2 < first.metaLength(); d2++) {
            if(buf.length() > 0) {
              buf.append(", ");
            }
            if(first.data(i, d2) == null) {
              buf.append("null");
            }
            else {
              buf.append(first.data(i, d2));
            }
          }
          logger.warning("null value in joined data, row " + i + " column " + d + FormatUtil.NEWLINE + "[" + buf.toString() + "]");
          break;
        }
      }
    }

    return first;
  }

  @Override
  protected Logging getLogger() {
    return logger;
  }

  /**
   * Parameterization class.
   * 
   * @author Erich Schubert
   * 
   * @apiviz.exclude
   */
  public static class Parameterizer extends AbstractDatabaseConnection.Parameterizer {
    /**
     * The static option ID
     */
    public static final OptionID SOURCES_ID = OptionID.getOrCreateOptionID("join.sources", "The data sources to join.");

    /**
     * The data souces to use.
     */
    protected List<DatabaseConnection> sources;

    @Override
    protected void makeOptions(Parameterization config) {
      super.makeOptions(config);
      super.configFilters(config);
      final ObjectListParameter<DatabaseConnection> sourcesParam = new ObjectListParameter<DatabaseConnection>(SOURCES_ID, DatabaseConnection.class);
      if(config.grab(sourcesParam)) {
        sources = sourcesParam.instantiateClasses(config);
      }
    }

    @Override
    protected LabelJoinDatabaseConnection makeInstance() {
      return new LabelJoinDatabaseConnection(filters, sources);
    }
  }
}