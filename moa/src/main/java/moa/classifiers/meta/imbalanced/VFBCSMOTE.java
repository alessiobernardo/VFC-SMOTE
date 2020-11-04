/*
 *    RebalanceStream.java
 * 
 *    @author Alessio Bernardo (alessio dot bernardo at polimi dot com)
 *    @author Emanuele Della Valle (emanuele dot dellavalle at polimi dot com)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package moa.classifiers.meta.imbalanced;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserverHistogram;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserverHistogram;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.trees.HoeffdingAdaptiveTreeHistogram;
import moa.classifiers.trees.HoeffdingTreeHistogram.ActiveLearningNode;
import moa.classifiers.trees.HoeffdingTreeHistogram.FoundNode;
import moa.classifiers.trees.HoeffdingTreeHistogram.Node;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;
import java.util.ArrayList;
import java.util.Random;


/**
 * VFBCSMOTE
 *
 * <p>
 * This strategy saves all the mis-classified samples in a histogram managed by ADWIN.
 * There is also the possibility to save a percentage of correctly classified instances.  
 * In the meantime, a model is trained with the data in input. 
 * When the minority sample ratio is less than a certain threshold, an ONLINE BORDERLINE SMOTE version is applied.
 * A random minority sample is chosen from the histogram and a new synthetic sample is generated 
 * until the minority sample ratio is greater or equal than the threshold.
 * The model is then trained with the new samples generated.
 </p>
 *
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classifier to train. Default is ARF</li>
 * <li>-h : Histogram to use. Default is HATHistogram, a modified version of HAT</li>
 * <li>-t : Threshold for the minority samples. Default is 0.5</li>
 * <li>-z : Percentage of correctly classified instances to save into the histrogram. Default is 0.0 </li>
 * <li>-m : Minimum number of samples in the minority class for applying SMOTE. Default is 100</li>
 * <li>-d : Should use ADWIN as drift detector? If enabled it is used by the method 
 * 	to track the performance of the classifiers and adapt when a drift is detected.</li>
 * </ul>
 *
 * @author Alessio Bernardo (alessio dot bernardo at polimi dot com) 
 * @version $Revision: 1 $
 */
public class VFBCSMOTE extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "OnlineSMOTE strategy that saves the data in a sliding window and when the minority class ratio is less than a threshold it generates some synthetic new samples using SMOTE";
    }
    
    private static final long serialVersionUID = 1L;
    
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "meta.AdaptiveRandomForest");        
    
    public ClassOption baseHistogramOption = new ClassOption("baseHistogram", 'h',
            "Histrogram to use.", Classifier.class, "trees.HoeffdingAdaptiveTreeHistogram");        
    
    /*
    public IntOption neighborsOption = new IntOption("neighbors", 'k',
            "Number of neighbors for SMOTE.",
            5, 1, Integer.MAX_VALUE); 
    */
    
    public FloatOption thresholdOption = new FloatOption("threshold", 't',
            "Minority class samples threshold.",
            0.0, 0.0, 1.0); 
    
    
    public FloatOption percentageCorrectlyClassifiedOption = new FloatOption("percentageCorrectlyClassified", 'z',
            "Percentage of instances correctly classied to save.",
            0.5, 0.0, 1.0);     
    
    public IntOption minSizeAllowedOption = new IntOption("minSizeAllowed", 'm',
            "Minimum number of samples in the minority class for appling SMOTE.",
            100, -1, Integer.MAX_VALUE); 
    
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'd',
            "Should use ADWIN as drift detector?");           
    
    protected Classifier learner; 
    protected HoeffdingAdaptiveTreeHistogram histrogram;
 
    protected double threshold;
    protected double percentageCorrectlyClassified;
    protected int minSizeAllowed;
    protected int nCorrectlyClassified;
    protected int nCorrectlySaved;
    protected boolean driftDetection;
            
    protected DoubleVector generatedClassDistribution;  
    protected ArrayList<Integer> alreadyUsed = new ArrayList<Integer>(); 
    protected ADWIN driftDetector;
    
    protected SamoaToWekaInstanceConverter samoaToWeka = new SamoaToWekaInstanceConverter();
    protected WekaToSamoaInstanceConverter wekaToSamoa = new WekaToSamoaInstanceConverter();    
	protected int[] indexValues;

    
    @Override
    public void resetLearningImpl() {     	    	
        this.learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);         
        this.histrogram = (HoeffdingAdaptiveTreeHistogram) getPreparedClassOption(this.baseHistogramOption);          
        this.threshold = this.thresholdOption.getValue();
        this.percentageCorrectlyClassified = this.percentageCorrectlyClassifiedOption.getValue();        
        this.minSizeAllowed = this.minSizeAllowedOption.getValue();
        this.driftDetection = !this.disableDriftDetectionOption.isSet();
        this.learner.resetLearning();            
        this.generatedClassDistribution = new DoubleVector();
      	this.alreadyUsed.clear();      	      	   
      	this.indexValues = null;
      	this.driftDetector = new ADWIN();
      	this.nCorrectlyClassified = 0;
      	this.nCorrectlySaved = 0;
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
    	double[] prediction = this.learner.getVotesForInstance(instance);    	
        return prediction;
    }
    
    @Override
    public void trainOnInstanceImpl(Instance instance) {     	
    	this.learner.trainOnInstance(instance);
    	if (Utils.maxIndex(this.learner.getVotesForInstance(instance)) != instance.classValue()) {
    		this.histrogram.trainOnInstance(instance);
    	} 
    	//save percentageCorrectlyClassified of the others
    	else {
    		if (this.percentageCorrectlyClassified != 0.0) {
    			this.nCorrectlyClassified ++;
    			double perc = this.percentageCorrectlyClassified * this.nCorrectlyClassified;
    			int toSave = (int) perc;
    			if (toSave > this.nCorrectlySaved) {
    				this.nCorrectlySaved ++;
    				this.histrogram.trainOnInstance(instance);
    			}    			
    		}    		    		
    	}
    	    	
    	//drift detection
		if (this.driftDetection) {
			driftDetection(instance);
		}
		
    	//check if the number of minority class samples are greater than -m
		boolean allowSMOTE = false;
		if (this.histrogram.observedClassDistribution.numValues() == instance.classAttribute().numValues()) {
			if (this.minSizeAllowed != -1) {
				int nMinClass = (int) this.histrogram.observedClassDistribution.minWeight();							
				if (nMinClass > this.minSizeAllowed) {
					allowSMOTE = true;
				}
			} else {
				allowSMOTE = true;
			}			
		}									
		//found leaves
		FoundNode[] leaves = this.histrogram.getLeaves(null, -1, this.minSizeAllowed);
		//found real minority class
		int minClass = 0;
		if ((this.histrogram.observedClassDistribution.getValue(0) + this.generatedClassDistribution.getValue(0)) <= 
				(this.histrogram.observedClassDistribution.getValue(1) + this.generatedClassDistribution.getValue(1))) {
			minClass = 0;
		} else {
			minClass = 1;
		}		
		//Apply the online SMOTE version until the ratio will be equal to the threshold			
		while (this.threshold > calculateRatio() && allowSMOTE && leaves.length != 0) {											
			Instance newInstance = generateNewInstance(minClass,leaves,instance);    		
    		this.generatedClassDistribution.addToValue(minClass, 1);
			this.learner.trainOnInstance(newInstance);			
		} 
		this.alreadyUsed.clear();	
    }
 
    private double calculateRatio() {
    	double ratio = 0.0;
    	//class 0 is the real minority
		if ((this.histrogram.observedClassDistribution.getValue(0) + this.generatedClassDistribution.getValue(0)) <= (this.histrogram.observedClassDistribution.getValue(1) + this.generatedClassDistribution.getValue(1))) {
			ratio = ( (double) this.histrogram.observedClassDistribution.getValue(0) + (double) this.generatedClassDistribution.getValue(0) ) / 
					( (double) this.histrogram.observedClassDistribution.getValue(0) + (double) this.generatedClassDistribution.getValue(0) + (double) this.histrogram.observedClassDistribution.getValue(1) + (double) this.generatedClassDistribution.getValue(1));			
		}
		//class 1 is the real minority
		else {
			ratio = ( (double) this.histrogram.observedClassDistribution.getValue(1) + (double) this.generatedClassDistribution.getValue(1) ) / 
					( (double) this.histrogram.observedClassDistribution.getValue(0) + (double) this.generatedClassDistribution.getValue(0) + (double) this.histrogram.observedClassDistribution.getValue(1) + (double) this.generatedClassDistribution.getValue(1));			
		}    					
    	return ratio;
    }	   
    
    private Instance generateNewInstance(int minClass, FoundNode[] leaves, Instance instance) {       	    	    		    	
    	//find randomly an instance
    	Random rand = new Random(1);
		int pos = rand.nextInt(leaves.length);
        while (this.alreadyUsed.contains(pos)) {
        	pos = rand.nextInt(leaves.length);
        }
        this.alreadyUsed.add(pos);
        if (this.alreadyUsed.size() == leaves.length) {
        	this.alreadyUsed.clear();
        }	
        FoundNode leaf = leaves[pos];        
        Node leafNode = leaf.node;
        if (leafNode == null) {        	
    		leafNode = leaf.parent;                               
        }            
        
        double[] values = new double[instance.numAttributes()];
        for (int i = 0; i < ((ActiveLearningNode) leafNode).getAttributeObservers().size(); i++) {  
        	int instAttIndex = modelAttIndexToInstanceAttIndex(i, instance);
            AttributeClassObserver obs = ((ActiveLearningNode) leafNode).getAttributeObservers().get(i);
            if (obs != null) {
            	if (obs instanceof GaussianNumericAttributeClassObserverHistogram) {
            		if (((GaussianNumericAttributeClassObserverHistogram) obs).getAttValDistPerClass().get(minClass) == null) {
            			values[instAttIndex] = 0;
            		} else {
            			double mean = ((GaussianNumericAttributeClassObserverHistogram) obs).getAttValDistPerClass().get(minClass).getSimpleMean();
                    	double variance = ((GaussianNumericAttributeClassObserverHistogram) obs).getAttValDistPerClass().get(minClass).getSimpleVariance();            	
                    	values[instAttIndex] = rand.nextGaussian()*variance+mean;
            		}                	
                }
                else if (obs instanceof NominalAttributeClassObserverHistogram) {
                	if (((NominalAttributeClassObserverHistogram) obs).attValDistPerClassSimple.get(minClass) == null) {                		
                		values[instAttIndex] = rand.nextInt(instance.attribute(instAttIndex).numValues());
                	} else {
                		values[instAttIndex] = ((NominalAttributeClassObserverHistogram) obs).attValDistPerClassSimple.get(minClass).maxIndex();
                	}                	            	
                }
            } else {
            	if (instance.attribute(instAttIndex).isNominal()) {
            		values[instAttIndex] = rand.nextInt(instance.attribute(instAttIndex).numValues());
            	} else {
            		values[instAttIndex] = 0;            		
            	}            	
            }
            
        }
        values[instance.classIndex()] = minClass;
        
        if (this.indexValues == null) {    		
    		this.indexValues = new int[instance.numAttributes()];
    		for (int i = 0; i < instance.numAttributes(); i ++) {
    			this.indexValues[i] = i;
    		}
    	}  
        
        //new synthetic instance
		Instance synthetic = instance.copy();
		synthetic.addSparseValues(this.indexValues, values, instance.numAttributes());		
		
		return synthetic;
    	    	
    }

    private void driftDetection(Instance instance) {
    	double pred = Utils.maxIndex(this.learner.getVotesForInstance(instance));
		double errorEstimation = this.driftDetector.getEstimation();
		double inputValue = pred == instance.classValue() ? 1.0 : 0.0;
		boolean resInput = this.driftDetector.setInput(inputValue);
		if (resInput) {
			if (this.driftDetector.getEstimation() > errorEstimation) {							        			        	
        		this.learner.resetLearning();
        		this.driftDetector = new ADWIN();		        	
			}
		}
    }
    
    @Override
    public boolean isRandomizable() {
    	if (this.learner != null) {
    		return this.learner.isRandomizable();	
    	}
    	else {
    		return false;
    	}
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }
    
    public String toString() {
        return "SMOTE online stategy using " + this.learner + " and ADWIN as sliding window";
    }       
    
}
