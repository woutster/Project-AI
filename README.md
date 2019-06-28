# Project-AI

In order to battle climate change, electric busses (should) become more of a common sight on our roads. However, the so called range-anxiety is real: people are scared their electric vehicles will not get them to their destination on one charge. ViriCiti is a company that aims to make electric vehicles, specifically electric (city) busses, a more common sight on our roads and also make them drive as efficient as possible. 
            
What influences the energy consumption of vehicles a lot is breaking and accelerating. When a bus is stuck in a traffic jam, the energy consumption of the bus will be higher than if it can just cruise to the next stop. Therefore, traffic flow is an interesting concept to be able to predict. 
            
This project aimed to predict traffic flow on set pieces of road using a combination of geofencing and neural nets. The suspicion is that the days of the week, hour of the day, and previous traffic flow play a big part in predicting the coming traffic flow. These features were taken into account in training the models.

# Running the code

All the code needed to read in the files, create new files (if using new busses), and train the LSTM can be found above. `main.py` will run all necessary scripts. The geofence (found in `get_gps.py` is fixed to Utrecht, but can be changed accordingly. 
            