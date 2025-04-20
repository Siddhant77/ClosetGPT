from icalendar import Calendar, IncompleteComponent
import recurring_ical_events
from pathlib import Path
import datetime
import weather

# Function to extract itinerary from an iCalendar file (located at ics_path) for the given date
# Returns a list of events (sorted by start time), represented as dictionaries with the following keys
#   "name": name of the event
#   "dtstart", "dtend": start/end of the event, as datetime objects
#   [OPTIONAL] "description", "location": description/location of the event
def get_itinerary(ics_path: str, date: datetime.date):
    itinerary = []

    calendar = Calendar.from_ical(Path(ics_path).read_bytes())
    events = recurring_ical_events.of(calendar, skip_bad_series=True).at(date)

    for event in events:
        dct = {}

        # Name is optional, set empty if missing
        try:
            dct["name"] = str(event["SUMMARY"])
        except KeyError:
            dct["name"] = ""

        # Start/end times are required
        try:
            dct["dtstart"] = event.start
            dct["dtend"] = event.end
        except IncompleteComponent:
            continue
        
        # Description is optional
        try:
            dct["description"] = str(event["DESCRIPTION"])
        except KeyError:
            pass

        # Location is optional
        try:
            dct["location"] = str(event["LOCATION"])
        except KeyError:
            pass

        itinerary.append(dct)
    
    itinerary.sort(key=lambda dct : dct["dtstart"])
    return itinerary

# Function to get both itinerary (from provided .ics path) and weather info for the current day at the given location (default is College Station)
# Returns a dictionary with the following keys
#   "weather": dictionary with weather info, as returned by weather.get_weather()
#   "itinerary": list with event names from itinerary (sorted by start time)
def get_itinerary_and_weather(ics_path: str, latitude: float=30.628, longitude: float=-96.3344, timezone: str="America/Chicago"):
    itin = get_itinerary(ics_path=ics_path, date=datetime.date.today())
    if len(itin) == 0:
        return {"weather": weather.get_weather(latitude=latitude, longitude=longitude, timezone=timezone),
                "itinerary": []}
    
    itin_names = [event["name"] for event in itin]
    itin_start = min([event["dtstart"] for event in itin])
    itin_end = max([event["dtend"] for event in itin])
    start_hour = itin_start.hour
    end_hour = itin_end.hour    # inclusive

    # Just in case itinerary extends to next day
    if itin_end.date() > itin_start.date():
        end_hour = 23

    return {"weather": weather.get_weather(latitude=latitude, longitude=longitude, timezone=timezone, start=start_hour, end=end_hour+1),
            "itinerary": itin_names}
