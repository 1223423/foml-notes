MON = read.csv("prediction_MON_clean.csv")
TUE = read.csv("prediction_TUE_clean.csv")
WED = read.csv("prediction_WED_clean.csv")
THU = read.csv("prediction_THU_clean.csv")
FRI = read.csv("prediction_FRI_clean.csv")
SAT = read.csv("prediction_SAT_clean.csv")
SUN = read.csv("prediction_SUN_clean.csv")

library(dplyr)

MON <- MON %>%
  group_by(prediction) %>%
  summarise(n()
  )

MON
