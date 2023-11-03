from typing import Any, Mapping, Sequence, Tuple, Union
from fmbase.source.merra2.model import MERRA2DataInterface
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from  fmbase.source.merra2.base import MERRA2Base
from fmbase.util.config import cfg
import pandas as pd
import numpy as np
import xarray as xa

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.
TargetLeadTimes = Union[TimedeltaLike, Sequence[TimedeltaLike], slice]  # slice with TimedeltaLike as its start and stop.

"""
    Configuration parameters from cfg().task:
    
    input_duration: pandas.Timedelta or something convertible to it (e.g. a
      shorthand string like '6h' or '5d12h').
    target_lead_times: Either a single lead time, a slice with start and stop
      (inclusive) lead times, or a sequence of lead times. Lead times should be
      Timedeltas (or something convertible to). They are given relative to the
      final input timestep, and should be positive.
"""

class FMGDataManager(MERRA2Base):
    SEC_PER_HOUR = 3600
    HOUR_PER_DAY = 24
    SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY
    AVG_DAY_PER_YEAR = 365.24219
    AVG_SEC_PER_YEAR = SEC_PER_DAY * AVG_DAY_PER_YEAR
    DAY_PROGRESS = "day_progress"
    YEAR_PROGRESS = "year_progress"

    def __init__(self):
        MERRA2Base.__init__(self)

    @classmethod
    def get_year_progress(cls, seconds_since_epoch: np.ndarray) -> np.ndarray:
      """Computes year progress for times in seconds.

      Args:
        seconds_since_epoch: Times in seconds since the "epoch" (the point at which UNIX time starts).

      Returns:
        Year progress normalized to be in the [0, 1) interval for each time point.
      """

      years_since_epoch = ( seconds_since_epoch / cls.SEC_PER_DAY / np.float64(cls.AVG_DAY_PER_YEAR))
      return np.mod(years_since_epoch, 1.0).astype(np.float32)

    @classmethod
    def get_day_progress( cls, seconds_since_epoch: np.ndarray, longitude: np.ndarray ) -> np.ndarray:
      """Computes day progress for times in seconds at each longitude.

      Args:
        seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
          point at which UNIX time starts).
        longitude: 1D array of longitudes at which day progress is computed.

      Returns:
        2D array of day progress values normalized to be in the [0, 1) inverval
          for each time point at each longitude.
      """

      # [0.0, 1.0) Interval.
      day_progress_greenwich = ( np.mod(seconds_since_epoch, cls.SEC_PER_DAY) / cls.SEC_PER_DAY )

      # Offset the day progress to the longitude of each point on Earth.
      longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
      day_progress = np.mod( day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 )
      return day_progress.astype(np.float32)


    @classmethod
    def featurize_progress( cls, name: str, dims: Sequence[str], progress: np.ndarray ) -> Mapping[str, xa.DataArray]:
      """Derives features used by ML models from the `progress` variable.

      Args:
        name: Base variable name from which features are derived.
        dims: List of the output feature dimensions, e.g. ("day", "lon").
        progress: Progress variable values.

      Returns:
        Dictionary of xarray variables derived from the `progress` values. It
        includes the original `progress` variable along with its sin and cos
        transformations.

      Raises:
        ValueError if the number of feature dimensions is not equal to the number
          of data dimensions.
      """
      if len(dims) != progress.ndim:
         raise ValueError( f"Number of feature dimensions ({len(dims)}) must be equal to the number of data dimensions: {progress.ndim}." )
      progress_phase = progress * (2 * np.pi)
      return {
          name: xa.DataArray(progress, dims=dims, ),
          name + "_sin": xa.DataArray( np.sin(progress_phase), dims=dims ),
          name + "_cos": xa.DataArray( np.cos(progress_phase), dims=dims ),
      }


    def add_derived_vars(self, data: xa.DataArray ) -> Dict[str,xa.DataArray]:
      """Constructs year and day progress features.
      """
      longitude_coord = data.coords["x"]
      time_coord = data.coords["time"]
      progress_features = {}

      # Compute seconds since epoch.
      # Note `data.coords["datetime"].astype("datetime64[s]").astype(np.int64)`
      # does not work as xarrays always cast dates into nanoseconds!
      seconds_since_epoch = (  time_coord.data.astype("datetime64[s]").astype(np.int64)  )

      # Add year progress features.
      year_progress: np.ndarray = self.get_year_progress(seconds_since_epoch)
      progress_features.update( self.featurize_progress( name=self.YEAR_PROGRESS, dims= ("time",), progress=year_progress ) )

      # Add day progress features.

      day_progress: np.ndarray =  self.get_day_progress(seconds_since_epoch, longitude_coord.data)
      progress_features.update( self.featurize_progress( name=self.DAY_PROGRESS, dims=("time","x"), progress=day_progress ) )

      return progress_features

    def extract_input_target_times( self, dataset: xa.Dataset ) -> Tuple[xa.Dataset, xa.Dataset]:
      """Extracts inputs and targets for prediction, from a Dataset with a time dim.

      The input period is assumed to be contiguous (specified by a duration), but
      the targets can be a list of arbitrary lead times.

      Examples:

        # Use 18 hours of data as inputs, and two specific lead times as targets:
        # 3 days and 5 days after the final input.
        extract_inputs_targets(
            dataset,
            input_duration='18h',
            target_lead_times=('3d', '5d')
        )

        # Use 1 day of data as input, and all lead times between 6 hours and
        # 24 hours inclusive as targets. Demonstrates a friendlier supported string
        # syntax.
        extract_inputs_targets(
            dataset,
            input_duration='1 day',
            target_lead_times=slice('6 hours', '24 hours')
        )

        # Just use a single target lead time of 3 days:
        extract_inputs_targets(
            dataset,
            input_duration='24h',
            target_lead_times='3d'
        )

      Args:
        dataset: An xa.Dataset with a 'time' dimension whose coordinates are
          timedeltas. It's assumed that the time coordinates have a fixed offset /
          time resolution, and that the input_duration and target_lead_times are
          multiples of this.

      Returns:
        inputs:
        targets:
          Two datasets with the same shape as the input dataset except that a
          selection has been made from the time axis, and the origin of the
          time coordinate will be shifted to refer to lead times relative to the
          final input timestep. So for inputs the times will end at lead time 0,
          for targets the time coordinates will refer to the lead times requested.
      """

      input_duration: TimedeltaLike = cfg().task.input_duration
      (target_lead_times, target_duration) = self.process_target_lead_times_and_get_duration()

      # Shift the coordinates for the time axis so that a timedelta of zero
      # corresponds to the forecast reference time. That is, the final timestep
      # that's available as input to the forecast, with all following timesteps
      # forming the target period which needs to be predicted.
      # This means the time coordinates are now forecast lead times.
      time = dataset.coords["time"]
      dataset = dataset.assign_coords(time=time + target_duration - time[-1])

      # Slice out targets:
      targets = dataset.sel({"time": target_lead_times})

      input_duration = pd.Timedelta(input_duration)
      # Both endpoints are inclusive with label-based slicing, so we offset by a
      # small epsilon to make one of the endpoints non-inclusive:
      zero = pd.Timedelta(0)
      epsilon = pd.Timedelta(1, "ns")
      inputs = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
      return inputs, targets

    @classmethod
    def process_target_lead_times_and_get_duration(cls) -> TimedeltaLike:
      """Returns the minimum duration for the target lead times."""
      target_lead_times: TargetLeadTimes = cfg().task.target_lead_times
      if isinstance(target_lead_times, slice):
        # A slice of lead times. xarray already accepts timedelta-like values for
        # the begin/end/step of the slice.
        if target_lead_times.start is None:
          # If the start isn't specified, we assume it starts at the next timestep
          # after lead time 0 (lead time 0 is the final input timestep):
          target_lead_times = slice(  pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step )
        target_duration = pd.Timedelta(target_lead_times.stop)
      else:
        if not isinstance(target_lead_times, (list, tuple, set)):
          # A single lead time, which we wrap as a length-1 array to ensure there
          # still remains a time dimension (here of length 1) for consistency.
          target_lead_times = [target_lead_times]

        # A list of multiple (not necessarily contiguous) lead times:
        target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
        target_lead_times.sort()
        target_duration = target_lead_times[-1]
      return target_lead_times, target_duration


    def extract_inputs_targets_forcings( self, data_vars: Dict[str,xa.DataArray] ) -> Tuple[xa.Dataset, xa.Dataset, xa.Dataset]:
      input_vars: List[str] = cfg().task.input_variables
      target_vars: List[str] = cfg().task.target_variables
      forcing_vars: List[str] = cfg().task.forcing_variables

      data_vars.update( self.add_derived_vars( data_vars.values()[0] ) )

      inputs, targets = self.extract_input_target_times( dataset )

      if set(forcing_vars) & set(target_vars):
        raise ValueError( f"Forcing variables {forcing_vars} should not overlap with target variables {target_vars}." )

      inputs = inputs[list(input_vars)]
      # The forcing uses the same time coordinates as the target.
      forcings = targets[list(forcing_vars)]
      targets = targets[list(target_vars)]

      return inputs, targets, forcings

    def load_training_data(self,**kwargs) -> Tuple[xa.Dataset, xa.Dataset, xa.Dataset]:
        input_vars: List[str] = cfg().task.input_variables
        forcing_vars: List[str] = cfg().task.forcing_variables
        years: List[int] = list(range(*cfg().task.year_range))
        months: List[int] = list(range(*cfg().task.month_range))
        varlist: List[str] = input_vars + forcing_vars
        levels: List[float] = list(cfg().task.levels)
        training_data: Dict[str,xa.DataArray] = {}

        for vname in varlist:
            vslices: List[xa.DataArray] = []
            for year in years:
                for month in months:
                    varray: xa.DataArray = self.load_cache_var( vname, year, month, **kwargs  )
                    vslices.append( varray.sel(z=levels) )
            training_data[vname] = xa.concat( vslices, dim="time" )

        return self.extract_inputs_targets_forcings( training_data )
