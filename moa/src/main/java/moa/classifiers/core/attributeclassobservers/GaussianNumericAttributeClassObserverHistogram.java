/*
 *    GaussianNumericAttributeClassObserver.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.core.attributeclassobservers;

import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;
import moa.core.Utils;

import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.GammaDistribution;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.AttributeSplitSuggestionHistrogram;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;

import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.GaussianEstimatorHistogram;
import moa.options.AbstractOptionHandler;
import com.github.javacliparser.IntOption;

/**
 * Class for observing the class data distribution for a numeric attribute using gaussian estimators.
 * This observer monitors the class distribution of a given attribute.
 * Used in naive Bayes and decision trees to monitor data statistics on leaves.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class GaussianNumericAttributeClassObserverHistogram extends AbstractOptionHandler
        implements NumericAttributeClassObserver {

    private static final long serialVersionUID = 1L;

    protected DoubleVector minValueObservedPerClass = new DoubleVector();

    protected DoubleVector maxValueObservedPerClass = new DoubleVector();

    protected AutoExpandVector<GaussianEstimatorHistogram> attValDistPerClass = new AutoExpandVector<GaussianEstimatorHistogram>();        

    public IntOption numBinsOption = new IntOption("numBins", 'n',
            "The number of bins.", 10, 1, Integer.MAX_VALUE);
    
    public AutoExpandVector<GaussianEstimatorHistogram> getAttValDistPerClass() {
    	return this.attValDistPerClass;
    }
    
    public DoubleVector getMinValueObservedPerClass() {
    	return this.minValueObservedPerClass;
    }
    
    public DoubleVector getMaxValueObservedPerClass() {
    	return this.maxValueObservedPerClass;
    }
    
    public double getSampleFromBeta(int minClass) { //qui
    	//double meanScaled = (this.attValDistPerClass.get(minClass).getSimpleMean() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass));
		//double varianceScaled = Math.pow(((this.attValDistPerClass.get(minClass).getSimpleStdDev() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass))),2);
    	double meanScaled = (this.attValDistPerClass.get(minClass).getMean() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass));
		double varianceScaled = Math.pow(((this.attValDistPerClass.get(minClass).getStdDev() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass))),2);
		double alpha = Math.pow(meanScaled, 2) * (((1-meanScaled) / varianceScaled) - (1 / meanScaled));
		double beta = alpha * ((1 / meanScaled) - 1);						
		return (this.minValueObservedPerClass.getValue(minClass) + ((this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass)) * new BetaDistribution(alpha,beta).sample()));
    }
    
    public double getSampleFromGamma(int minClass) { //qui
    	double meanScaled = (this.attValDistPerClass.get(minClass).getSimpleMean() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass));
		double varianceScaled = Math.pow(((this.attValDistPerClass.get(minClass).getSimpleStdDev() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass))),2);
    	//double meanScaled = (this.attValDistPerClass.get(minClass).getMean() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass));
		//double varianceScaled = Math.pow(((this.attValDistPerClass.get(minClass).getStdDev() - this.minValueObservedPerClass.getValue(minClass)) / (this.maxValueObservedPerClass.getValue(minClass) - this.minValueObservedPerClass.getValue(minClass))),2);
		double alpha = Math.pow(meanScaled, 2) / varianceScaled;
		double beta = 1;						
		return new GammaDistribution(alpha,beta).sample();
    }

    @Override
    public void observeAttributeClass(double attVal, int classVal, double weight) {
        if (Utils.isMissingValue(attVal)) {
        } else {
        	GaussianEstimatorHistogram valDist = this.attValDistPerClass.get(classVal);
            if (valDist == null) {
                valDist = new GaussianEstimatorHistogram();
                this.attValDistPerClass.set(classVal, valDist);
                this.minValueObservedPerClass.setValue(classVal, attVal);
                this.maxValueObservedPerClass.setValue(classVal, attVal);
            } else {
                if (attVal < this.minValueObservedPerClass.getValue(classVal)) {
                    this.minValueObservedPerClass.setValue(classVal, attVal);
                }
                if (attVal > this.maxValueObservedPerClass.getValue(classVal)) {
                    this.maxValueObservedPerClass.setValue(classVal, attVal);
                }
            }
            valDist.addObservation(attVal, weight);
        }
    }

    @Override
    public double probabilityOfAttributeValueGivenClass(double attVal,
            int classVal) {
    	GaussianEstimatorHistogram obs = this.attValDistPerClass.get(classVal);
        return obs != null ? obs.probabilityDensity(attVal) : 0.0;
    }

    @Override
    public AttributeSplitSuggestion getBestEvaluatedSplitSuggestion(
            SplitCriterion criterion, double[] preSplitDist, int attIndex,
            boolean binaryOnly) {
        AttributeSplitSuggestion bestSuggestion = null;
        double[] suggestedSplitValues = getSplitPointSuggestions();
        for (double splitValue : suggestedSplitValues) {
            double[][] postSplitDists = getClassDistsResultingFromBinarySplit(splitValue);
            double merit = criterion.getMeritOfSplit(preSplitDist,
                    postSplitDists);
            if ((bestSuggestion == null) || (merit > bestSuggestion.merit)) {
                bestSuggestion = new AttributeSplitSuggestion(
                        new NumericAttributeBinaryTest(attIndex, splitValue,
                        true), postSplitDists, merit);
            }
        }
        return bestSuggestion;
    }
    
    @Override
    public AttributeSplitSuggestionHistrogram getBestEvaluatedSplitSuggestionHistogram(
            SplitCriterion criterion, double[] preSplitDist, int attIndex,
            boolean binaryOnly) {
    	AttributeSplitSuggestionHistrogram bestSuggestion = null;
        double[] suggestedSplitValues = getSplitPointSuggestions();
        for (double splitValue : suggestedSplitValues) {
            double[][] postSplitDists = getClassDistsResultingFromBinarySplit(splitValue);
            double[][] postSplitDistsNode = getClassDistsResultingFromBinarySplitHistogram(splitValue);
            double merit = criterion.getMeritOfSplit(preSplitDist,
                    postSplitDists);
            if ((bestSuggestion == null) || (merit > bestSuggestion.merit)) {
                bestSuggestion = new AttributeSplitSuggestionHistrogram(
                        new NumericAttributeBinaryTest(attIndex, splitValue,
                        true), postSplitDists,postSplitDistsNode, merit);
            }
        }
        return bestSuggestion;
    }

    public double[] getSplitPointSuggestions() {
        Set<Double> suggestedSplitValues = new TreeSet<Double>();
        double minValue = Double.POSITIVE_INFINITY;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.attValDistPerClass.size(); i++) {
        	GaussianEstimatorHistogram estimator = this.attValDistPerClass.get(i);
            if (estimator != null) {
                if (this.minValueObservedPerClass.getValue(i) < minValue) {
                    minValue = this.minValueObservedPerClass.getValue(i);
                }
                if (this.maxValueObservedPerClass.getValue(i) > maxValue) {
                    maxValue = this.maxValueObservedPerClass.getValue(i);
                }
            }
        }
        if (minValue < Double.POSITIVE_INFINITY) {
            double range = maxValue - minValue;
            for (int i = 0; i < this.numBinsOption.getValue(); i++) {
                double splitValue = range / (this.numBinsOption.getValue() + 1.0) * (i + 1)
                        + minValue;
                if ((splitValue > minValue) && (splitValue < maxValue)) {
                    suggestedSplitValues.add(splitValue);
                }
            }
        }
        double[] suggestions = new double[suggestedSplitValues.size()];
        int i = 0;
        for (double suggestion : suggestedSplitValues) {
            suggestions[i++] = suggestion;
        }
        return suggestions;
    }

    // assume all values equal to splitValue go to lhs
    public double[][] getClassDistsResultingFromBinarySplit(double splitValue) {
        DoubleVector lhsDist = new DoubleVector();
        DoubleVector rhsDist = new DoubleVector();
        for (int i = 0; i < this.attValDistPerClass.size(); i++) {
        	GaussianEstimatorHistogram estimator = this.attValDistPerClass.get(i);
            if (estimator != null) {
                if (splitValue < this.minValueObservedPerClass.getValue(i)) {
                    rhsDist.addToValue(i, estimator.getTotalWeightObserved());
                } else if (splitValue >= this.maxValueObservedPerClass.getValue(i)) {
                    lhsDist.addToValue(i, estimator.getTotalWeightObserved());
                } else {
                    double[] weightDist = estimator.estimatedWeight_LessThan_EqualTo_GreaterThan_Value(splitValue);
                    lhsDist.addToValue(i, weightDist[0] + weightDist[1]);
                    rhsDist.addToValue(i, weightDist[2]);
                }
            }
        }
        return new double[][]{lhsDist.getArrayRef(), rhsDist.getArrayRef()};
    }
    
    // assume all values equal to splitValue go to lhs
    public double[][] getClassDistsResultingFromBinarySplitHistogram(double splitValue) {
        DoubleVector lhsDist = new DoubleVector();
        DoubleVector rhsDist = new DoubleVector();
        for (int i = 0; i < this.attValDistPerClass.size(); i++) {
        	GaussianEstimatorHistogram estimator = this.attValDistPerClass.get(i);
            if (estimator != null) {
                if (splitValue < this.minValueObservedPerClass.getValue(i)) {
                    rhsDist.addToValue(i, estimator.getTotalInstancesObserved());
                } else if (splitValue >= this.maxValueObservedPerClass.getValue(i)) {
                    lhsDist.addToValue(i, estimator.getTotalInstancesObserved());
                } else {
                    double[] weightDist = estimator.estimatedWeight_LessThan_EqualTo_GreaterThan_ValueHistogram(splitValue);
                    lhsDist.addToValue(i, weightDist[0] + weightDist[1]);
                    rhsDist.addToValue(i, weightDist[2]);
                }
            }
        }
        
        return new double[][]{roundValues(lhsDist.getArrayRef()), roundValues(rhsDist.getArrayRef())};
    }
    
    protected double[] roundValues(double[] dist) {
    	for (int i = 0; i < dist.length; i++) {
    		dist[i] = Math.round(dist[i]);
        }
    	return dist;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        // TODO Auto-generated method stub
    }

    @Override
    public void observeAttributeTarget(double attVal, double target) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
