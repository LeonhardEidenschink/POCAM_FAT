import numpy as np
from copy import deepcopy
def decode_value(value, 
                ):
    """
    value is 10 word hex string
    """
    dt_coarse = 1/175*1e3 # in ns
    dt_fine = 1/350/8*1e3
    value = deepcopy(value)
    if len(value)==12 and value[0:2] == "0x":
        value = value.replace("0x", "")
    value = bin(int(value, base=16))[2:].zfill(40)
    #
    channel = value[-37-1:-35-1]
    edge_bit = value[-33-1]
    #
    fine_time = value[-2-1:]
    phase_bit = value[-3-1]
    coarse_time = value[-32-1:-4]
    epoch_rollover = value[-35-1]
    #
    decode_error = value[-34-1]
    time_ns = int(coarse_time, base=2)*dt_coarse + int(fine_time, base=2)*dt_fine + dt_coarse/2 * int(phase_bit=='1')
    return (int(channel,base=2), int(edge_bit), time_ns, int(epoch_rollover))

def get_timestamps(in_):
    dtype=[('ch', int),
       ('edge', int),
       ('time_ns', float), 
       ('epoch_ro', int), 
       ('epoch', int)
      ]

    out_ = np.zeros(len(in_), dtype=dtype)
    for i_ in range(len(in_)):
        out_[i_] = decode_value(in_[i_])+(0,)
    out_['epoch'] = np.cumsum(out_['epoch_ro'])
    out_['time_ns'] += (2**29/0.175 ) * out_['epoch']
    return out_[out_['epoch_ro']==0]

def getADCreadings(in_):
    dtype=[('base', int),
       ('peak', int),
       ('sum', int), 
      ]
    out_ = np.zeros(in_.shape, dtype=dtype)
    for i_ in range(len(in_)):
        out_['base'][i_] = int( '0x' + in_[i_].decode()[-4:], base=16)
        out_['peak'][i_] = int( '0x' + in_[i_].decode()[-8:-4], base=16)
        out_['sum'][i_]  = int( '0x' + in_[i_].decode()[-16:-8], base=16)
    return out_

def convert_to_perTrig(indata, ntrigs = None, timerange = (-200, 100000)):
    if ntrigs is None:
        ntrigs = indata['tdc3'].shape[0]/2
        if not ntrigs.is_integer():
            raise RuntimeError("extracted number of triggers is not integer")
        ntrigs=int(ntrigs)
    else:
        if not (indata['tdc3'].shape == (ntrigs*2,)):
            raise RuntimeError(f"Requested {ntrigs} triggers, but there are {indata['tdc3'].shape[0] } edges in timing channel")
    outdata = {}
    outdata['tdc0'] = [] 
    outdata['tdc1'] = []
    outdata['tdc2'] = []
    outdata['tdc3'] = np.zeros( (ntrigs, 2), dtype = indata['tdc3'].dtype )
    for i_ in range(0, ntrigs):
        outdata['tdc3'][i_] = indata['tdc3'][2*i_:2*i_+2]
        for fifo_ in ['tdc0', 'tdc1', 'tdc2']:
            selbool_ = ( ( (indata[fifo_]['time_ns'] - outdata['tdc3']['time_ns'][i_, 0] ) > timerange[0] )*
                         ( (indata[fifo_]['time_ns'] - outdata['tdc3']['time_ns'][i_, 1] ) < timerange[1] )
                       )
            outdata[fifo_].append(np.array(indata[fifo_][selbool_]))
            del selbool_
    for fifo_ in ['adcA','adcB']:
        if not (indata[fifo_].shape == (ntrigs, )):
            raise RuntimeError(f"Error! Fifo {fifo_} expects {ntrigs} triggers, but shape is {indata[fifo_].shape}")
        outdata[fifo_]= np.array(indata[fifo_])
    return(outdata)



def extract_per_pwm(dataset):
    fifo_list = [] 
    for k_ in list(dataset.keys()):
        if k_.endswith("_nentr"):
            fifo_list.append(k_.split("_nentr")[0])
    del(k_)
    
    # we need to convert back (here and everywhere else it is necessary) the values/types of all the entries
    # of the several data arrays that had to be changed to be able to store them inside an hdf5 file.
    # they were basically all converted to 'S' strings, but we have to be aware if they were floats, int or hex numbers 
    # (or others) initially to properly convert them back 
    
    pwms = np.array( [int(b.decode('utf-8')) for b in dataset['fl_pwm'][()] ] )
    #print('000: ',pwms)
    data = {}
    inds = {}
    for fifo_ in fifo_list:
        inds[fifo_] = np.array([0] + list(np.cumsum( np.array( [ int(b.decode('utf-8')) for b in dataset[f'{fifo_}_nentr'] ] ) ) ) )
        #print('001: ', inds, inds[fifo_])
    #print(inds)
    for i_,pwm_ in enumerate(pwms):
        data[pwm_] = {}
        for fifo_ in fifo_list:
            if fifo_ in ['tdc0', 'tdc1', 'tdc2','tdc3']:
                data[pwm_][fifo_] = get_timestamps(dataset[f'{fifo_}_vals'][inds[fifo_][i_]:inds[fifo_][i_+1]])
                #print('002: ', data[pwm_][fifo_])
            else:
                data[pwm_][fifo_] = getADCreadings(dataset[f'{fifo_}_vals'][inds[fifo_][i_]:inds[fifo_][i_+1]])
        data[pwm_] = convert_to_perTrig(data[pwm_] )
        data[pwm_].update(get_SiPM_dt(data[pwm_]))
    return(data)



def get_SiPM_dt(data):
    out_ = {}
    for fifo_  in ['tdc0', 'tdc1', 'tdc2', 'tdc3']:
        if not fifo_ in data.keys(): 
            continue
        out_[fifo_+"_dt"] = np.zeros(len(data[fifo_]))
        for i_ in range(len(out_[fifo_+"_dt"])):
            if len(data[fifo_][i_])< 2:
                continue
            rise_bool = data[fifo_][i_]['edge'] == 1
            fall_bool = data[fifo_][i_]['edge'] == 0
            if np.sum(fall_bool) < 1 :
                continue
            if np.sum(rise_bool) < 1 :
                continue      
            out_[fifo_+"_dt"][i_] = (data[fifo_][i_]['time_ns'][fall_bool][0] - data[fifo_][i_]['time_ns'][rise_bool][0])
    return(out_)


'''
def extract_file_values(f_, style="new"):
    fifo_list = ['tdc0', 'tdc1', 'tdc2', 'tdc3', 'adcA', 'adcB']
    data_extracted = {}
    indices = {}
    metadata = {'ntrigs': f_['metadata']['ntrigs'][()],
                'pwm_values': f_['pwm_values'][()], 
                "trigPer": f_['metadata']['trigPer'][()],
                "thr_tdc0": f_['metadata']['thr_tdc0'][()],
                "thr_tdc1": f_['metadata']['thr_tdc1'][()],
                "thr_tdc2": f_['metadata']['thr_tdc2'][()],
               }
    for fifo_ in fifo_list:
        metadata[f'{fifo_}_nentr'] =  f_[f'{fifo_}_nentr'][()]
        indices[fifo_] = np.array([0]+np.cumsum(f_[f'{fifo_}_nentr'][()]).tolist())
    for i_, pwm_ in enumerate(metadata['pwm_values']):
        data_extracted[pwm_] = {}
        for fifo_ in fifo_list:
            data_extracted[pwm_][fifo_] = [l_.decode() for l_ in f_[f'{fifo_}_vals'][ indices[fifo_][i_]:indices[fifo_][i_+1]]]
        for ch_ in ["A", "B"]:
            data_extracted[pwm_][f'adc{ch_}_base'] = np.zeros(metadata['ntrigs'],dtype=int)
            data_extracted[pwm_][f'adc{ch_}_peak'] = np.zeros(metadata['ntrigs'],dtype=int)
            data_extracted[pwm_][f'adc{ch_}_sum'] = np.zeros(metadata['ntrigs'],dtype=int)
            for j_ in range(0, metadata['ntrigs']):
                if style=="legacy":
                    data_extracted[pwm_][f'adc{ch_}_sum'][j_] = int('0x' + data_extracted[pwm_][f'adc{ch_}'][2*j_ +1][-8:], base=16)
                    data_extracted[pwm_][f'adc{ch_}_base'][j_] = int( '0x'+data_extracted[pwm_][f'adc{ch_}'][2*j_][-4:], base=16)
                    data_extracted[pwm_][f'adc{ch_}_peak'][j_] = int( '0x'+data_extracted[pwm_][f'adc{ch_}'][2*j_][-8:-4], base=16)
                elif style=="new":
                    data_extracted[pwm_][f'adc{ch_}_base'][j_] = int( '0x'+data_extracted[pwm_][f'adc{ch_}'][j_][-4:], base=16)
                    data_extracted[pwm_][f'adc{ch_}_peak'][j_] = int( '0x'+data_extracted[pwm_][f'adc{ch_}'][j_][-8:-4], base=16)
                    data_extracted[pwm_][f'adc{ch_}_sum'][j_] =  int( '0x'+data_extracted[pwm_][f'adc{ch_}'][j_][-16:-8], base=16)
                else: 
                    raise RuntimeError("Unknown data file style")
        for tdc_ in ['tdc0', 'tdc1', 'tdc2', 'tdc3']:
            cur_times_ = get_timestamps(data_extracted[pwm_][tdc_])
            if np.sum(cur_times_['ch'] != int(tdc_.split('tdc')[-1]))>0:
                raise RuntimeError("Wrong TDC channel")
            
            if (not (len(cur_times_['time_ns']) == 2*metadata['ntrigs']) ) and (tdc_ == 'tdc3') :
                print("Warning! TDC3 channels has number of entriels not equal 2x n_triggers")
            data_extracted[pwm_][tdc_+"_time_ns"] = np.array(cur_times_['time_ns'])
            data_extracted[pwm_][tdc_+"_edge"] = np.array(cur_times_['edge'])
        data_extracted[pwm_]['tdc3_per_trig_time'] = np.zeros((metadata['ntrigs'], 2))
        data_extracted[pwm_]['tdc3_per_trig_time'][:,0] = data_extracted[pwm_]["tdc3_time_ns"][0::2]
        data_extracted[pwm_]['tdc3_per_trig_time'][:,1] = data_extracted[pwm_]["tdc3_time_ns"][1::2]
        data_extracted[pwm_]['tdc3_tot'] =  data_extracted[pwm_]['tdc3_per_trig_time'][:,1] -data_extracted[pwm_]['tdc3_per_trig_time'][:,0]
        ###
        data_extracted[pwm_]['tdc3_per_trig_edge'] = np.zeros((metadata['ntrigs'], 2),dtype=int)
        data_extracted[pwm_]['tdc3_per_trig_edge'][:,0] = data_extracted[pwm_]["tdc3_edge"][0::2]
        data_extracted[pwm_]['tdc3_per_trig_edge'][:,1] = data_extracted[pwm_]["tdc3_edge"][1::2]
        ###
        for tdc_ in ['tdc0', 'tdc1', 'tdc2']:
            data_extracted[pwm_][tdc_ + "_per_trig_time"]=[]
            data_extracted[pwm_][tdc_ + "_per_trig_edge"]=[]
            data_extracted[pwm_][tdc_ + "_tot"] = np.nan*np.ones(metadata['ntrigs'])
            for k_ in range(metadata['ntrigs']):
                dt_ = data_extracted[pwm_][tdc_+'_time_ns'] - data_extracted[pwm_]['tdc3_per_trig_time'][k_,0]
                selbool_ = (dt_ > -100)*(dt_ < 100000)
                data_extracted[pwm_][tdc_ + "_per_trig_time"].append(data_extracted[pwm_][tdc_+'_time_ns'][selbool_])
                data_extracted[pwm_][tdc_ + "_per_trig_edge"].append(data_extracted[pwm_][tdc_+'_edge'][selbool_])
                if len(data_extracted[pwm_][tdc_ + "_per_trig_time"][k_])>1:
                    rise_bool_ = (data_extracted[pwm_][tdc_ + "_per_trig_edge"][k_]==1)
                    fall_bool_ = (data_extracted[pwm_][tdc_ + "_per_trig_edge"][k_]==0)
                    if np.any(rise_bool_) and np.any(fall_bool_)>0:
                        start_ = data_extracted[pwm_][tdc_ + "_per_trig_time"][k_][rise_bool_][0]
                        stop_ = data_extracted[pwm_][tdc_ + "_per_trig_time"][k_][fall_bool_][0]
                        data_extracted[pwm_][tdc_ + "_tot"][k_] = stop_-start_
                    else: 
                        data_extracted[pwm_][tdc_ + "_tot"][k_]=0.0
                else: 
                    data_extracted[pwm_][tdc_ + "_tot"][k_]=0.0
    return {'metadata': metadata, 
            'values': data_extracted}
'''
