/*
 *    WindowClassificationPerformanceEvaluator.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet@cs.waikato.ac.nz)
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
package moa.evaluation;

import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;

import com.github.javacliparser.IntOption;

/**
 * Classification evaluator that updates evaluation results using a sliding
 * window.
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @author Jean Paul Barddal (jpbarddal@gmail.com)
 * @version $Revision: 8 $
 *
 *
 */
public class WindowFixedClassificationPerformanceEvaluator extends BasicClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;

    public IntOption widthOption = new IntOption("width",
            'w', "Size of Window", 1000);
    
    public IntOption widthGradualDriftOption = new IntOption("GradualDriftWidth",
            'j', "Size of Window", -1);

    @Override
    protected Estimator newEstimator() {
        return new WindowEstimator(this.widthOption.getValue(), this.widthGradualDriftOption.getValue());
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == WindowFixedClassificationPerformanceEvaluator.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    public class WindowEstimator implements Estimator {

        protected double[] window;

        protected int posWindow;

        protected int lenWindow;

        protected int SizeWindow;

        protected double sum;

        protected double qtyNaNs;
        
        protected int SizeGDWindow;

        public WindowEstimator(int sizeWindow, int sizeGD) {
            window = new double[sizeWindow];
            SizeWindow = sizeWindow;
            posWindow = 0;
            lenWindow = 0;
            SizeGDWindow = sizeGD;
        }

        public void add(double value) {
        	if (SizeGDWindow != -1) {
        		//case of gradual cd -> 1 window of SizeWindow length, 2 window of SizeGDWindow lenght, 3 window of SizeWindow length and so on
        		if (posWindow == SizeWindow && window.length == SizeWindow) {
                    posWindow = 0;
                    window = new double[SizeGDWindow];
                    sum = 0;
                    qtyNaNs = 0;
                    lenWindow = 0;
                }        	
            	else if (posWindow == SizeGDWindow && window.length == SizeGDWindow) {
                    posWindow = 0;
                    window = new double[SizeWindow];
                    sum = 0;
                    qtyNaNs = 0;
                    lenWindow = 0;
                }
        	} else {
        		//normal case
        		if (posWindow == SizeWindow) {
                    posWindow = 0;
                    window = new double[SizeWindow];
                    sum = 0;
                    qtyNaNs = 0;
                    lenWindow = 0;
                } 
        	}
        	
        	
            if(!Double.isNaN(value)) {
                sum += value;
            }else qtyNaNs++;
            
            window[posWindow] = value;
            posWindow++;
            lenWindow++;
            
            
        }

        public double estimation(){
            return sum / (lenWindow - qtyNaNs);
        }

    }

}
