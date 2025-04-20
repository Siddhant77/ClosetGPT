import openmeteo_requests
from openmeteo_requests.Client import OpenMeteoRequestsError

import requests_cache
from retry_requests import retry

WEATHER_CODES = {
	0: "clear",
	1: "mostly clear",
	2: "partly cloudy",
	3: "overcast",
	45: "foggy",
	48: "rime",
	51: "light drizzle",
	53: "moderate drizzle",
	55: "dense drizzle",
	56: "light freezing drizzle",
	57: "dense freezing drizzle",
	61: "slight rain",
	63: "moderate rain",
	65: "heavy rain",
	66: "light freezing rain",
	67: "heavy freezing rain",
	71: "slight snowfall",
	73: "moderate snowfall",
	75: "heavy snowfall",
	77: "snow grains",
	80: "slight rain showers",
	81: "moderate rain showers",
	82: "violent rain showers",
	85: "slight snow showers",
	86: "heavy snow showers",
	95: "thunderstorm",
	96: "thunderstorm with slight hail",
	99: "thunderstorm with heavy hail"
}

# Function to get basic weather info for the current day at the given location (default is College Station)
# If start and end indices are provided, hourly metrics are restricted to that range of hours
# Returns a dictionary with the following keys
	# "status": 0 if the API request failed, otherwise 1
	# "code": qualitative description of the prevalent weather condition for the day (not super accurate)
	# "temp_high", "temp_low", "temp_avg": high/low/average hourly temperatures (Fahrenheit)
	# "feelslike_high", "feelslike_low", "feelslike_avg": high/low/average hourly feels-like temperatures (Fahrenheit)
	# "precip_chance_max", "precip_chance_min", "precip_chance_avg": max/min/average hourly precipitation chances (percentage)
	# "precip_total": total precipitation for the given range (inches)
def get_weather(latitude: float=30.628, longitude: float=-96.3344, timezone: str="America/Chicago", start: int=0, end: int=24):
	assert(24 >= end > start >= 0)
	summary = {}

	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": ["temperature_2m", "apparent_temperature", "precipitation_probability", "precipitation"],
		"daily": "weather_code",
		"timezone": timezone,
		"forecast_days": 1,
		"temperature_unit": "fahrenheit",
		"precipitation_unit": "inch"
	}
	try:
		response = openmeteo.weather_api(url, params=params)[0]
	except OpenMeteoRequestsError as inst:
		print("Error with API request:")
		print(inst.args[0]["reason"])

		summary["status"] = 0
		return summary

	# Process hourly data for the range start:end. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()[start:end]
	hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()[start:end]
	hourly_precipitation_probability = hourly.Variables(2).ValuesAsNumpy()[start:end]
	hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()[start:end]

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_weather_code = daily.Variables(0).ValuesAsNumpy()

	# Extract key metrics
	summary["code"] = WEATHER_CODES[daily_weather_code[0]]
	summary["temp_high"] = max(hourly_temperature_2m)
	summary["temp_low"] = min(hourly_temperature_2m)
	summary["temp_avg"] = sum(hourly_temperature_2m) / len(hourly_temperature_2m)
	summary["feelslike_high"] = max(hourly_apparent_temperature)
	summary["feelslike_low"] = min(hourly_apparent_temperature)
	summary["feelslike_avg"] = sum(hourly_apparent_temperature) / len(hourly_apparent_temperature)
	summary["precip_chance_max"] = max(hourly_precipitation_probability)
	summary["precip_chance_min"] = min(hourly_precipitation_probability)
	summary["precip_chance_avg"] = sum(hourly_precipitation_probability) / len(hourly_precipitation_probability)
	summary["precip_total"] = sum(hourly_precipitation)

	summary["status"] = 1
	print(summary)
	return summary
