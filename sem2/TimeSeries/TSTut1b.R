#LECTURE 1
#CARBON DIOXIDE 
#A SAFE WAY TO IMPORT DATA: MAUNA LOA EXAMPLE
www = "https://www.mimuw.edu.pl/~noble/courses/TimeSeries/data/atmospheric-carbon-dioxide-recor.csv"
carbon = read.csv(www)
carbon = carbon[-611,]
y = carbon$MaunaLoaCO2
#MAKE A TIME SERIES OBJECT WITH PERIOD 12 (MONTHS)
MaunLoaCo2 = ts(data = y, frequency = 12)
?stl
output.stl = stl(MaunLoaCo2, s.window = "periodic")
plot(output.stl)

a <- output.stl$time.series
acf(a)
apply(a,2,sd)

#AIR PASSENGER DATA
data(AirPassengers)
AP <- AirPassengers
str(AP)

?HoltWinters

AP.hw <- HoltWinters(AP,seasonal="mult")
plot(AP.hw)
legend("topleft",c("observed","fitted"),lty=1,col=1:2)
AP.predict <-predict(AP.hw,n.ahead=4*12)
ts.plot(AP,AP.predict,lty=1:2)

#TUTORIAL 1
#EXAMPLES OF TIME PLOTS: WE USE PACKAGE fpp2
library(fpp2)
#(NOTE THIS PACKAGE ALSO NEEDS ggplot2, fma, forecast and expsmooth)
#EXPONENTIAL SMOOTHING IS ANOTHER TERM FOR HOLT WINTERS
#WE WANT TO ILLUSTRATE THE autoplot COMMAND WHICH IS USEFUL
autoplot(melsyd[,"Economy.Class"]) +
        ggtitle("Economy class passengers: Melbourne-Sydney") + 
      xlab("Year") +
       ylab("Thousands")
#THE ggseasonplot IS VERY USEFUL - FOR EACH YEAR IT PLOTS THE MONTHLY PROGRESSION
ggseasonplot(a10, year.labels=TRUE, year.labels.left=TRUE) +
       ylab("$ million") +
       ggtitle("Seasonal plot: antidiabetic drug sales")
ggseasonplot(a10, polar=TRUE) +
      ylab("$ million") +
       ggtitle("Polar seasonal plot: antidiabetic drug sales")
#A POLAR SEASON PLOT PUTS THE MONTHS AROUND A CIRCLE.
#A SUBSERIES PLOT WILL GIVE YOU THE TIME SERIES PLOTS FOR EACH MONTH
ggsubseriesplot(a10) +
      ylab("$ million") +
       ggtitle("Seasonal subseries plot: antidiabetic drug sales")
#NOW USE AUTOPLOT FOR A BIVARIATE TIME SERIES: ELECTICITY DEMAND
# AND THE RECORDED TEMPERATURE, USING AUTOPLOT
autoplot(elecdemand[,c("Demand","Temperature")], facets=TRUE) +
  xlab("Year: 2014") + ylab("") +
  ggtitle("Half-hourly electricity demand: Victoria, Australia")
qplot(Temperature, Demand, data=as.data.frame(elecdemand)) +
  ylab("Demand (GW)") + xlab("Temperature (Celsius)")
autoplot(visnights[,1:5], facets=TRUE) +
  ylab("Number of visitor nights each quarter (millions)")
#GGally IS A USEFUL PACKAGE - WITH A MULTIVARIATE TIME SERIES
#WE CAN CONSIDER IT PAIRWISE
library(GGally)
GGally::ggpairs(as.data.frame(visnights[,1:5]))
#NOW FOR THE EXAMPLE OF BEER CONSUMPTION AND LAGGED PLOTS
#WHICH INDICATE SEASONALITY.
beer2 <- window(ausbeer, start=1992)
gglagplot(beer2)
#THE Auto Correlation is in line with the information from the
#lagged plots
ggAcf(beer2)
#NOW CONSIDER MONTHLY ELECTRICITY DEMAND
aelec <- window(elec, start=1980)
autoplot(aelec) + xlab("Year") + ylab("GWh")
ggAcf(aelec, lag=48)
