import json, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'latent_history_config.json').read_text())

def sat_vp_pa(t_c):
    t=np.asarray(t_c,dtype=float)
    return 610.78*np.exp(17.2694*t/(t+238.3))

def _factor_value(hist, key, spec_key):
    code=int(hist[key])
    if code==0:
        if spec_key=='C_radiation_multiplier':
            return 1.0
        return 0.0
    spec=CFG['factors'][spec_key]
    return float(spec['low'] if code<0 else spec['high'])

def get_history(history_id):
    if history_id=='HNR':
        raise ValueError('HNR is a native-reset reference and has no generated prehistory')
    if history_id=='H00':
        return dict(CFG['histories']['H00'])
    for h in CFG['histories']['corners']:
        if h['id']==history_id:
            return dict(h)
    raise KeyError(history_id)

def generate_prehistory(history_id, duration_h=72, end_local_hour=8.0):
    hist=get_history(history_id)
    n=int(round(float(duration_h)*3600/CFG['common_grid_s']))
    elapsed=np.arange(n+1,dtype=float)*float(CFG['common_grid_s'])
    local_hour=np.mod(float(end_local_hour)-float(duration_h)+elapsed/3600.0,24.0)

    base=CFG['base_diurnal']
    rad=np.zeros_like(local_hour)
    day=(local_hour>=base['radiation_day_start_hour'])&(local_hour<=base['radiation_day_end_hour'])
    rad[day]=base['radiation_peak_w_m2']*np.sin(
        np.pi*(local_hour[day]-base['radiation_day_start_hour'])/
        (base['radiation_day_end_hour']-base['radiation_day_start_hour'])
    )
    phase=2*np.pi*(local_hour-base['outdoor_temperature_phase_zero_hour'])/24.0
    tout=base['outdoor_temperature_mean_c']+base['outdoor_temperature_amplitude_c']*np.sin(phase)
    rh=base['outdoor_rh_mean_pct']-base['outdoor_rh_amplitude_pct']*np.sin(phase)
    co2=np.full_like(local_hour,base['outdoor_co2_ppm'],dtype=float)
    soil=np.full_like(local_hour,base['soil_boundary_temperature_c'],dtype=float)

    if history_id=='H00':
        t_off=0.0; rh_off=0.0; r_mult=1.0; co2_off=0.0
    else:
        t_off=_factor_value(hist,'A_thermal','A_thermal_offset_c')
        rh_off=_factor_value(hist,'B_humidity','B_rh_offset_pct')
        r_mult=_factor_value(hist,'C_radiation','C_radiation_multiplier')
        co2_off=_factor_value(hist,'D_co2','D_co2_offset_ppm')

    tout=tout+t_off
    rh=np.clip(rh+rh_off,*CFG['factors']['B_rh_offset_pct']['clip'])
    rad=np.maximum(0.0,rad*r_mult)
    co2=co2+co2_off
    soil=soil+CFG['factors']['A_thermal_offset_c']['soil_fraction']*t_off
    sky=tout+base['sky_temperature_offset_c']
    vp=sat_vp_pa(tout)*rh/100.0

    return pd.DataFrame({
      'elapsed_s':elapsed,
      'local_hour':local_hour,
      'global_radiation_w_m2':rad,
      'outdoor_temperature_c':tout,
      'outdoor_rh_pct':rh,
      'outdoor_vapor_pressure_pa':vp,
      'outdoor_co2_ppm':co2,
      'wind_speed_m_s':np.full_like(local_hour,base['wind_speed_m_s'],dtype=float),
      'sky_temperature_c':sky,
      'soil_boundary_temperature_c':soil,
    })

if __name__=='__main__':
    out=HERE/'generated_anchor_histories'
    out.mkdir(exist_ok=True)
    for hid in CFG['anchor_primary_history_ids']:
        generate_prehistory(hid,CFG['primary_prehistory_h'],CFG['anchor_end_local_hour']).to_csv(
            out/f'{hid}_72h.csv',index=False,float_format='%.12g'
        )
    print(f'generated {len(CFG["anchor_primary_history_ids"])} deterministic 72-h histories in {out}')
