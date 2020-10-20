# Used Car Price Prediction
Shift Academy - Final Project <br>
The dataset contains 6,019 observations (cars) incorporates 3,205 diesel cars, 2,746 petrol cars, 56 CNG cars, 10 LPG cars, and 2 electric cars in India.

This project is done in a team with members:
1. Abdillah Fikri
2. Kintan Pitaloka W
3. Rizky Muhammad Kahfie

## Data Structure
Data Fields <br>
There are id and 12 characteristics of cars and the price.

1. Name - The brand and model of the car.
2. Location - The location in which the car is being sold or is available for purchase.
3. Year - The year or edition of the model.
4. Kilometers_Driven - The total kilometers are driven in the car by the previous owner(s) in KM.
5. Fuel_Type - The type of fuel used by the car. (Petrol, Diesel, Electric, CNG, LPG)
6. Transmission - The type of transmission used by the car. (Automatic / Manual)
7. Owner_Type - First, Second, Third, or Fourth & Above
8. Mileage - The standard mileage offered by the car company in kmpl or km/kg
9. Engine - The displacement volume of the engine in CC.
10. Power - The maximum power of the engine in bhp.
11. Seats - The number of seats in the car.
12. Price - The price of the car (target).

## Repository Structure
1. Data <br>
    Contains raw data and processed data
   
2. Markdown <br>
    Contains markdown version of notebooks
   
3. Notebooks <br>
    Contains results of data analysis and processing in the form of jupyter notebook which is divided into 4 parts:
   
    * data_prep: Data preparation before visualization
    * data_vis: Data visualization and analysis
    * preprocessing: Preprocessing of data before modeling such as train test splits, missing values and outlier handling, categorical encoding, feature engineering, etc.
    * modeling: Model making
   
4. Outputs <br>
    Contains model result in * .pkl format
   
5. Scripts <br>
    Contains python script version of notebooks
