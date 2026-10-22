import numpy as np
from g1_pipeline.spectral_qa import velocity_spectrum


def test_spectrum_detects_above_target_nyquist_and_preserves_low_frequency():
    t=np.arange(1200)/120;values=np.stack([np.sin(2*np.pi*3*t),np.sin(2*np.pi*25*t)],axis=1)
    result=velocity_spectrum(values,120)
    assert result['flagged_coordinates']==[1]
    assert result['high_band_power_fraction'][0]<.001
    assert result['high_band_power_fraction'][1]>.99
