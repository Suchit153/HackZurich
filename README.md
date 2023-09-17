# EcoTrack - HackZurich 2023

![devpost thumbnail ecotrack](https://github.com/Suchit153/HackZurich/assets/73829218/f2276d24-c6ab-4d6a-a7fb-246efbb82b02)

## ✨ Inspiration
We have had many painful experiences going through traffic jams on Swiss roads. In this hackathon, we wanted to think of a sustainable solution that could relieve the suffering of the driver and the passenger(s) or a way to reduce the congestion. Introducing: **EcoTrack** 🛣️

## ❓ What it does
Our app helps to motivate drivers to change their behavior by awarding them with scratch cards, coupons, and other benefits to pick a more sustainable route or slow down to reduce congestion.
It has multiple features:
- **`Traffic flow detector`**: Users can insert their starting point and destination, then our system will predict if there'll be a traffic jam ahead
- **`Departure time recommender`**: Select the planned departure time, then we can suggest ones with less congestion and a route that is environmentally friendly :)
- **`Prize inventory`**: Collect coupons, vouchers, and discounts by using the app and driving safely

## 💻 How we built it
- We developed a neural network model with `PyTorch` and `Scikit-Learn` to predict traffic flow based on BIT's data
- Tested different approaches and experimented with various error/loss functions to optimize our model
- To make the solution accessible to users, we deployed an online user interface (UI) using modern web and mobile development technologies

![image](https://github.com/Suchit153/HackZurich/assets/73829218/27beeb0d-d80d-4808-a91a-dd7ed284af91)

## ⚔️ Challenges we ran into
- Extracting and cleaning the huge amount of data was challenging
- We didn't have a lot of computation power or the time to go for stronger models
- No "obvious" solution - we had to be creative along the way

## 🏅 Accomplishments that we're proud of
- We managed to get a few hours of sleep
- Developed a novel framework to preprocess and clean the traffic data
- Successfully deployed an online web & mobile platform within less than 24 hours!
- Racked our brains in search of possible gamification elements in encouraging sustainable commuting habits. Previous ideas include: memes, ghosts to scare bad drivers, and VR plants that grow along your trip
- Achieved a 0.0000000049 mean squared error (MSE) loss on our own test dataset
- Made a nice video for our pitch ;)

https://github.com/Suchit153/HackZurich/assets/73829218/ecf133ca-38b2-4331-988d-d95ec60b7b13

## 📚 What we learned
- Split and assign the tasks early!
- How to preprocess and analyze large-scale traffic data efficiently
- The importance of user-centric design

## 🔮 What's next for EcoTrack
Winning HackZurich 2023

## 🤗 Meet our team!
- Viet Duc Kieu (_Vietnam_)
- Yohan Thibault (_France_)
- Sara Rutz (_Switzerland_)
- Suchit Gupta (_India_)
- Nathanya Queby (_Indonesia_)

## 🗣️ Fun fact
We all come from different countries and two of us took the highway to get to Zurich 🛣️
