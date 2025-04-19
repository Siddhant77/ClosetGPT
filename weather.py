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
# Returns a dictionary with the following keys
	# "status": 0 if the API request failed, otherwise 1
	# "code": qualitative description of the prevalent weather condition for the day
	# "temp_high", "temp_low", "temp_avg": high/low/average hourly temperatures (Fahrenheit)
	# "feelslike_high", "feelslike_low", "feelslike_avg": high/low/average hourly feels-like temperatures (Fahrenheit)
	# "precip_chance_max", "precip_chance_min", "precip_chance_avg": max/min/average hourly precipitation chances (percentage)
	# "precip_total": total precipitation for the day (inches)
def get_weather(latitude=30.628, longitude=-96.3344, timezone="America/Chicago"):
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
		"daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
			"apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
			"precipitation_probability_max", "precipitation_probability_min", "precipitation_probability_mean",
			"precipitation_sum"],
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

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_weather_code = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
	daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
	daily_apparent_temperature_max = daily.Variables(4).ValuesAsNumpy()
	daily_apparent_temperature_min = daily.Variables(5).ValuesAsNumpy()
	daily_apparent_temperature_mean = daily.Variables(6).ValuesAsNumpy()
	daily_precipitation_probability_max = daily.Variables(7).ValuesAsNumpy()
	daily_precipitation_probability_min = daily.Variables(8).ValuesAsNumpy()
	daily_precipitation_probability_mean = daily.Variables(9).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(10).ValuesAsNumpy()

	# Extract key metrics
	summary["code"] = WEATHER_CODES[daily_weather_code[0]]
	summary["temp_high"] = daily_temperature_2m_max[0]
	summary["temp_low"] = daily_temperature_2m_min[0]
	summary["temp_avg"] = daily_temperature_2m_mean[0]
	summary["feelslike_high"] = daily_apparent_temperature_max[0]
	summary["feelslike_low"] = daily_apparent_temperature_min[0]
	summary["feelslike_avg"] = daily_apparent_temperature_mean[0]
	summary["precip_chance_max"] = daily_precipitation_probability_max[0]
	summary["precip_chance_min"] = daily_precipitation_probability_min[0]
	summary["precip_chance_avg"] = daily_precipitation_probability_mean[0]
	summary["precip_total"] = daily_precipitation_sum[0]

	summary["status"] = 1
	print(summary)
	return summary
