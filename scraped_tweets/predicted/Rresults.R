MON = read.csv("prediction_MON_clean.csv")
TUE = read.csv("prediction_TUE_clean.csv")
WED = read.csv("prediction_WED_clean.csv")
THU = read.csv("prediction_THU_clean.csv")
FRI = read.csv("prediction_FRI_clean.csv")
SAT = read.csv("prediction_SAT_clean.csv")
SUN = read.csv("prediction_SUN_clean.csv")

MON_label = MON["prediction"]
TUE_label = TUE["prediction"]
WED_label = WED["prediction"]
THU_label = THU["prediction"]
FRI_label = FRI["prediction"]
SAT_label = SAT["prediction"]
SUN_label = SUN["prediction"]

df = data.frame()
MON_gr = select(MON, "prediction", "weekday_category")
TUE_gr = select(TUE, "prediction", "weekday_category")
WED_gr = select(WED, "prediction", "weekday_category")
THU_gr = select(THU, "prediction", "weekday_category")
FRI_gr = select(FRI, "prediction", "weekday_category")
SAT_gr = select(SAT, "prediction", "weekday_category")
SUN_gr = select(SUN, "prediction", "weekday_category")

df = Reduce(function(x, y) merge(x, y, all=TRUE), list(MON_gr, TUE_gr, WED_gr, THU_gr, FRI_gr, SAT_gr, SUN_gr))

x1 = sum(MON_gr[1])
MONDAY = c(x1, "Monday")
x2 = sum(TUE_gr[1])
TUESDAY = c(x2, "Tuesday")
x3 = sum(WED_gr[1])
WEDNESDAY = c(x3, "Wednesday")
x4 = sum(THU_gr[1])
THURSDAY = c(x4, "Thursday")
x5 = sum(FRI_gr[1])
FRIDAY = c(x5, "Friday")
x6 = sum(SAT_gr[1])
SATURDAY = c(x6, "Saturday")
x7 = sum(SUN_gr[1])
SUNDAY = c(x7, "Sunday")


lst = list(MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)

pos_count = c(MONDAY[1], TUESDAY[1], WEDNESDAY[1], THURSDAY[1], FRIDAY[1], SATURDAY[1], SUNDAY[1])
day = c(MONDAY[2], TUESDAY[2], WEDNESDAY[2], THURSDAY[2], FRIDAY[2], SATURDAY[2], SUNDAY[2])

finalforgraph = data.frame(day, count, stringsAsFactors = FALSE)