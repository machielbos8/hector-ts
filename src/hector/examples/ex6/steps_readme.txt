Information on database of potential steps in GPS time series

The steps master file can be found at http://geodesy.unr.edu/NGLStationPages/steps.txt
Steps pertaining to individual stations are additionally found in a table at the bottom of each station page.

In steps.txt:
Column 1 is the station 4-character ID

Column 2 is the step date in YYMMMDD format

Column 3 is the step type code where:
  Code=1 is time of an equipment change from IGS log file (antenna, receiver or firmware change)
  Code=2 is possible earthquake step where epicenter is within 10^(0.5*mag - 0.8) km of the station
  Code=3 is reserved for software or product changes that could introduce a discontinuity

if Code==1 or 3
  Column 4 is the type of equipment, software or product change event
if Code==2 it refers to an earthquake and
  Column 4 is the threshold distance for this event in km.
  Column 5 is the distance from station to epicenter in km.
  Column 6 is the event magnitude
  Column 7 is the USGS event ID. Event information available at http://earthquake.usgs.gov/earthquakes/eventpage/<eventID>

Potential earthquake related steps are marked if the distance from station to epicenter is less than the threshold distance calculated using a simple formula based on event magnitude: r0 = 10.^(0.5*mag - 0.79) km.
So a 4 gives 16 km
a 5 gives 51 km
a 6 gives 162 km
a 7 gives 512 km
an 8 gives 1622 km
a 9 gives 5129 km

Event depth, style or directionality of displacement is currently not accounted for in the distance threshold. Actual displacement may not have occurred at the marked time.

Information in the step file is updated daily using automated procedures.
