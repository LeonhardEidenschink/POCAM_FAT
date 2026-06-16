import os, sys
import time
from copy import deepcopy
import numpy as np 
from scipy.optimize import curve_fit



def exp_fit(x, rate, offset, lam):
    # function is overdefined on purpose
    return rate*np.exp(-lam*(x-offset))

def baseline(x, A, b1):
    return A*np.exp(b1*x) 

def ampl(x, b2, c2):
    return np.exp(b2*(x-c2) )
    
def breakdown_func(x, A, b1, b2, c2):
    return baseline(x, A, b1)  + ampl(x, b2, c2)


def get_breakdown_fit(xvals, yvals, yerrs, mode="pwm"):
    ### Getting baseline estimate
    params = {}
    params['pwm'] = {
        "base_below": 48000,
        "rise_above": 50000,
        "guess": (20, 0.0, 0.002, 50000),
        "bounds": ([0.00, 0.0, 0.0, 40000], 
                  [1e6, 10.0, 10.0, 60000])
    }
    params['V'] = {
        "base_below": 25,
        "rise_above": 28,
        "guess": (20, 0.0, 0.002, 27),
        "bounds": ([0.00, 0.0, 0.0, 20], 
                  [ 1e6, 10.0, 10.0, 35])
    }
    params = params[mode]
    selbool = xvals<params['base_below']
    x_ = xvals[selbool]
    y_ = yvals[selbool]
    err_ = yerrs[selbool]
    fit_base = curve_fit(baseline, x_, y_, sigma=err_, 
                        p0 = params['guess'][0:2],
                        bounds = (params['bounds'][0][0:2],
                                  params['bounds'][1][0:2])
                        )
    #print("base:", fit_base[0]) 
    #
    selbool = xvals>params['rise_above']
    x_ = xvals[selbool]
    y_ = yvals[selbool]
    err_ = yerrs[selbool]

    fit_slope = curve_fit(ampl, x_, y_, sigma=err_, 
                        p0 = params['guess'][2:4],
                        bounds = (params['bounds'][0][2:4],
                                  params['bounds'][1][2:4])
                                  )
    #print("slope: ", fit_slope[0])
    #### 
    full_fit = curve_fit(breakdown_func, xvals,yvals, sigma= yerrs, 
              p0 = (fit_base[0][0], fit_base[0][1],fit_slope[0][0], fit_slope[0][1]),
                bounds = params['bounds'] 
                           )
    
    return(full_fit)





class POCAMSiPMHandler:
    
    def __init__(self, session, target, verbose=False):
        self.s = session
        self.default_thresholds = {
            'tdc0': 50000,
            'tdc1': 50000,
            'tdc2': 50000,
        }
        self.search_range = {
            'tdc0': (23700, 35500),   ### 30000
            'tdc1': (24000, 35500)    ### 30000
            ## omitting tdc2 as not necessary
            }
        ## registers for thresholds and scalers
        self.regs_thr= {"tdc0": "$10", "tdc1": "$13", "tdc2": "$11"} 
        self.regs_scalers = {"tdc0": "$d0", "tdc1": "$d1", "tdc2": "$d2"}
        self.verbose=verbose
        self.target=target
        
        
    def find_threshold(self, tdc='', rate = 10.0, custom_thresholds = None, fine_step=3, 
                        reboot = True , sipm1_pwm = 0, sipm2_pwm = 0, hv='0'):
        
        found_scalers = False
        scalers_cntr = 0
        
        lower_range = self.search_range[tdc][0]
        upper_range = self.search_range[tdc][1]
        
        while (found_scalers == False) and (scalers_cntr < 5):
            scalers_cntr += 1
            
            if tdc=='':
                raise RuntimeError("::POCAMSiPMThresholdFinder:: Provide name of TDC: tdc0 or tdc1")
            elif not (tdc in self.regs_thr.keys()):
                raise RuntimeError(f"::POCAMSiPMThresholdFinder:: Unknown TDC: {tdc}")
            current_range = [lower_range, 
                             upper_range]
            if self.verbose:
                print(f"::POCAMSiPMThresholdFinder:: Doing search for {tdc} in pwm threshold range "
                      f"[{current_range[0]}, {current_range[1]}] for rate {rate } [ 1/s ]")

            # being within a factor of 2 is good enough, as precise value will be found in fine scan
            rate_target = [int(np.floor(0.5*rate)),int(np.ceil(2*rate)) ]
            if reboot:
                if self.verbose:
                    print("::POCAMSiPMThresholdFinder:: Doing resets first")
                self.s.toggle_boards(on=False)
                self.s.toggle_pwm(on=False)
                time.sleep(1)
                self.s.toggle_boards(on=True)
                self.s.toggle_pwm(on=True)

            self.s.pwm('sipm1', sipm1_pwm)
            self.s.pwm('sipm2', sipm2_pwm)

            #### updating thresholds
            self.enableTDCs(hv=hv)
            thresholds = deepcopy(self.default_thresholds)
            if not (custom_thresholds is None):
                for key_ in custom_thresholds.keys():
                    if not key_ in thresholds.keys():
                        raise ValueError(f"Requesting threshold for unknown TDC: {key_}")
                    thresholds[key_] = custom_thresholds[key_]
                    if self.verbose:
                        print("::POCAMSiPMThresholdFinder:: Custom threshold for TDC: {key_}: {thresholds[key_]}")
            for key_ in thresholds.keys():
                self.set_threshold(tdc=key_, value = thresholds[key_],verbose=self.verbose)
            ## let's try to find values at range first
            if self.verbose:
                self.s.info_pwm()
                print("::POCAMSiPMThresholdFinder::  Initial scan for search range....")

            scalers = {}
            for val_ in current_range:
                self.set_threshold(tdc=tdc, value = val_)
                time.sleep(3)
                scalers[val_] = self.get_scaler(tdc=tdc)
                if self.verbose:
                    print(f"\t threshold: {val_}, scaler: {scalers[val_]}")
            del val_  
            
            ### Making sanity checks at edges of the search range
            if scalers[current_range[0]] < rate_target[1]:
                print("::POCAMSiPMThresholdFinder:: WARNING! Rate at lower search range too low... Threshold might fail to find")
                
            if (scalers[current_range[1]] > rate_target[1]):
                found_scalers = False
                upper_range = upper_range + 1000
                print(f"Attempt No. {scalers_cntr} for POCAMSiPMThresholdFinder: Rate too high at upper range of threshold search -> Retry ")
                
            else:
                found_scalers = True
                
        if found_scalers == False: 
            raise RuntimeError("::POCAMSiPMThresholdFinder:: Error! Rate too high at upper range of threshold search")
            
                
        ### Now doing a classica divide-by-two range to get into target range
        found_guess = None
        n_iter = 0
        while (found_guess is None):
            val_ = int(np.mean(current_range))
            self.set_threshold(tdc=tdc, value = val_)
            time.sleep(3)
            scalers[val_] = self.get_scaler(tdc=tdc)
            print(f"\t threshold: {val_}, scaler: {scalers[val_]}", end='')            
            if scalers[val_]  < rate_target[0]:
                current_range[1] = val_
                print()
            elif scalers[val_]  > rate_target[1]:
                current_range[0] = val_
                print()
            else:
                found_guess=val_
                print("..... \t found guess")
            n_iter+=1
            if n_iter > 100:
                found_guess = -1
                print("Error! The threshold couldn't be found in 100 iterations")
        del val_
        print("----------- Doing fine scan  ----------------") 
        fine_thr = []
        fine_scaler = []
        for i_ in range(-7,8):
            val_= found_guess+i_*fine_step

            self.set_threshold(tdc=tdc, value=val_)
            time.sleep(3)
            scalers[val_]=self.get_scaler(tdc)
            print(f"\t threshold: {val_}, scaler: {scalers[val_]}")      
            fine_thr.append(int(val_))
            fine_scaler.append(scalers[val_])      
        del i_, val_
        ## Doing fit 
        fit_func = lambda x, offset, lam: rate*np.exp(-lam*(x-offset))
        xvals_ = np.array(fine_thr)
        yvals_ = np.array(fine_scaler)
        yerrs_ = np.sqrt(yvals_)
        yerrs_[yvals_==0] = 1.0
        fit_res = curve_fit(fit_func, xvals_, yvals_, sigma= yerrs_,
                            p0 = (1.*found_guess, 0.001),
                            bounds = ([10000, 0],[50000, 10])
                            )[0]
        #print(fit_res)
        all_thr = list(scalers.keys())
        all_thr.sort()
        # self.disableTDCs()
        for key_ in self.default_thresholds.keys():
            self.set_threshold(tdc=key_, value = thresholds[key_],verbose=True)
        ####
        result = {
            "tdc": deepcopy(tdc), 
            "all_thresholds": np.array(all_thr, dtype=int),
            "all_scalers": np.array([scalers[v_] for v_ in all_thr],dtype=int),
            "fine_thresholds":np.array(fine_thr,dtype=int),
            "fine_scalers": np.array(fine_scaler,dtype=int),
            "fit_result": deepcopy(fit_res),
            "thresholds" : deepcopy(thresholds),
            "target_rate": float(rate)
            }
        
        return result  
        
        
        
    def find_breakdown(self, tdc='tdc0', 
                       threshold = -1,
                       hv = '0',
                       sipm_pwms = None,
                       reboot = True):
        if threshold < 0 or threshold > 65535:
            raise ValueError("::POCAMSiPMBreakdownFinder:: Provide threshold in range betweel 0 and 65535")
        if not (tdc  in self.default_thresholds.keys()):
            raise ValueError(f"::POCAMSiPMBreakdownFinder:: Unknown TDC: {tdc}")
        if sipm_pwms is None:
            sipm_pwms=[]
            sipm_pwms+=list(np.linspace(32000, 45000, 16, dtype=int))  ### 16
            sipm_pwms+=list(np.linspace(45500, 56000, 17, dtype=int))  ### 17
            sipm_pwms.sort()
        if reboot:
            self.s.toggle_boards(on=False)
            self.s.toggle_pwm(on=False)
            time.sleep(1)
            self.s.toggle_boards(on=True)
            self.s.toggle_pwm(on=True)
        self.s.pwm('sipm1', 0)
        self.s.pwm('sipm2', 0)
        time.sleep(1)
        thresholds = self.default_thresholds
        for key_ in thresholds.keys():
            self.set_threshold(tdc=key_, value = thresholds[key_],verbose=True)
        ###
        hv_str = 'sipm{:d}'.format(int(hv)+1)
        print("HV name: ", hv_str)
        self.enableTDCs(hv=hv)
        self.set_threshold(tdc=tdc, value=threshold,verbose=True)
        data = {'scalers': [],
                'sipm_pwm':[],
                'sipm_V': []}
        for pwm_ in sipm_pwms:
            self.s.pwm(hv_str, pwm_)
            time.sleep(2.5)
            data['scalers'].append(self.get_scaler(tdc=tdc))            
            data['sipm_pwm'].append(pwm_)
            v_, v_raw_ = self.s.parseADC(self.s.adc_read(hv_str))
            data['sipm_V'].append(v_)
            print("Rate at {:d} ({:0.3f}V) = {:d}".format(
                data['sipm_pwm'][-1], data['sipm_V'][-1], data['scalers'][-1]))
        for k_ in data.keys():
            data[k_]=np.array(data[k_])
        ### 
        for m_ in ['pwm', 'V']:
            #print(m_)
            x_ = data[f'sipm_{m_}']
            if m_=='V':
                x_ = -1.*x_
            y_ = data['scalers']
            yerr_ = np.sqrt(y_)
            yerr_[y_==0] = 2.0
            data[f'fit_{m_}'] = get_breakdown_fit(
                            xvals=x_, yvals=y_, yerrs=yerr_, mode=m_)[0]
            data[f'breakdown_{m_}'] = (np.log(data[f'fit_{m_}'][0]) + data[f'fit_{m_}'][3]*data[f'fit_{m_}'][2])
            data[f'breakdown_{m_}'] /= (data[f'fit_{m_}'][2] -  data[f'fit_{m_}'][1])
        self.s.pwm('sipm1', 0)
        self.s.pwm('sipm2', 0)
        for key_ in self.default_thresholds.keys():
            self.set_threshold(tdc=key_, value = thresholds[key_],verbose=True)
        ###
        return(data)
    
    
    def set_threshold(self, tdc=None, value = -1, verbose=False):
        """
        This function sets threshold value for a given TDC
        """
        if verbose:
            print(f"::POCAMSiPMThresholdFinder:: Setting threshod {tdc} to {value}")
        if value < 0 or value > 65535:
            raise ValueError(f"::POCAMSiPMThresholdFinder:: Threshold value should be between 0 and 65535... got {value} for {tdc}")
        if not tdc in self.regs_thr.keys():
            raise ValueError(f"::POCAMSiPMThresholdFinder:: Unknown TDC value for threshold {tdc}")
        command_ = f"{self.target} {self.regs_thr[tdc]} 0 {str(int(value))} pcmSPI"
        if verbose:
            print("\tthreshold command : \'{:s}\'".format(command_)) 
        self.s.pcmCmd(command_)
        
        
    def get_scaler(self, tdc=''):
        if tdc=='':
            raise RuntimeError("::POCAMSiPMThresholdFinder:: Provide name of TDC: tdc0 or tdc1")
        elif not (tdc in self.regs_thr.keys()):
            raise RuntimeError(f"::POCAMSiPMThresholdFinder:: Unknown TDC for scaler: {tdc}")
        command_ = f"{self.target} {self.regs_scalers[tdc]} 0 0 pcmSPI"
        return int(self.s.pcmCmd(command_).split()[1], base=16)
    
    
    def enableTDCs(self, hv = '0'):
        self.s.set_sensors(target=self.target, 
                            enable5v=False, 
                            SipmHv = hv,
                            enableTdcBuffer=True, 
                            enableTdc0=True,
                            enableTdc1=True, 
                            enableTdc2=True,
                            enableTdc3=False, 
                            mode0='0',
                            mode1='0', 
                            enable0=True, 
                            enable1=False)
        
        
    def disableTDCs(self):
        self.s.set_sensors(target=self.target, 
                            enable5v=False, 
                            SipmHv = '-1',
                            enableTdcBuffer=False, 
                            enableTdc0=False,
                            enableTdc1=False, 
                            enableTdc2=False,
                            enableTdc3=False, 
                            mode0='0',
                            mode1='0', 
                            enable0=False, 
                            enable1=False)   
