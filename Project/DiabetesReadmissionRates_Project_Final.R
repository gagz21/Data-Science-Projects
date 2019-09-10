#installing necessary libraries
library(ggplot2)
library(plyr)
library(dplyr)
library(arules)
#set the working directory of the csv file
setwd("C:/Users/gagd2/Desktop/Syracuse/IST_707/Project")
#read and load the csv file
diabetesData <- read.csv("DiabeticReadmissionRates2.csv")
#look at the structure of the data
str(diabetesData)
#Data Cleaning - removing unnecessary attributes like patient id and encounter id
diabetesData <- diabetesData[,-1:-2]
#View the data to see if there are any more attributes that need to be removed.
View(diabetesData)
# Remove underscore from the columns and convert those to lower case
for (i in 1:ncol(diabetesData)) {
  names(diabetesData) <- tolower(gsub("_", "", names(diabetesData)))
}
# Taking a quick look at the dataset, attributes weight, payer_code have a lot of missing values.
#remove these attributes
diabetesData$weight <- NULL
diabetesData$payercode <- NULL
#diabetesData$medical_specialty <- NULL

# convert ) in age column to ]
diabetesData$age <- gsub(")", "]", diabetesData$age)

#Let's take a look to see if there are any more missing values in the remaining dataset.
diabetesData$race <- gsub("?",NA,diabetesData$race, fixed = TRUE)
diabetesData$medicalspecialty <- gsub("?",NA,diabetesData$medicalspecialty, fixed = TRUE)
diabetesData$diag1 <- gsub("?",NA,diabetesData$diag1, fixed = TRUE)
diabetesData$diag2 <- gsub("?",NA,diabetesData$diag2, fixed = TRUE)
diabetesData$diag3 <- gsub("?",NA,diabetesData$diag3, fixed = TRUE)

Total <-sum(is.na(diabetesData))
cat("The number of missing values in this data is ", Total )

# replace the missing values
diabetesData$race[which(is.na(diabetesData$race))] <- "Unknown"
diabetesData$medicalspecialty[which(is.na(diabetesData$medicalspecialty))] <- "Unknown"
diabetesData$diag1[which(is.na(diabetesData$diag1))] <- 0
diabetesData$diag2[which(is.na(diabetesData$diag2))] <- 0
diabetesData$diag3[which(is.na(diabetesData$diag3))] <- 0

diabetesData$readmitted=dplyr::recode(diabetesData$readmitted, ">30"=2, "<30"=1, NO=0)
diabetesData$medicalspecialty=dplyr::recode(diabetesData$medicalspecialty, "Unknown"="1",	"InternalMedicine"="2",
                                            "Family/GeneralPractice"="3",
                                            "Cardiology"="4",	"Surgery-General"="5",	"Orthopedics"="6",
                                            "Gastroenterology"="7",	"Nephrology"="8",	"Orthopedics-Reconstructive"="9",
                                            "Pulmonology"="10",	"Psychiatry"="11",	"Surgery-Neuro"="12",
                                            "Obsterics&Gynecology-GynecologicOnco"="13",	"Pediatrics-CriticalCare"="14",	"Endocrinology"="15",
                                            "Urology"="16","Radiology"="17",	"Pediatrics-Endocrinology"="18",
                                            "ObstetricsandGynecology"="19",	"Pediatrics"="20",	"Pediatrics-Hematology-Oncology"="21",
                                            "Surgery-Cardiovascular/Thoracic"="22",	"Anesthesiology-Pediatric"="23",	"Emergency/Trauma"="24",
                                            "Psychology"="25",	"Neurology"="26",	"Hematology/Oncology"="27",
                                            "Psychiatry-Child/Adolescent"="28",	"Surgery-Colon&Rectal"="29",	"Pediatrics-Pulmonology"="30",
                                            "Gynecology"="31",	"Pediatrics-Neurology"="32",	"Surgery-Plastic"="33",
                                            "Ophthalmology"="34",	"Surgery-Pediatric"="35",	"Pediatrics-EmergencyMedicine"="36",
                                            "PhysicalMedicineandRehabilitation"="37",	"Otolaryngology"="38",	"InfectiousDiseases"="39",
                                            "Podiatry"="40",	"Anesthesiology"="41",	"Oncology"="42",
                                            "Surgery-Maxillofacial"="43",	"Pediatrics-InfectiousDiseases"="44",	"Pediatrics-AllergyandImmunology"="45",
                                            "Surgeon"="46",	"Surgery-Vascular"="47",	"Osteopath"="48",
                                            "Surgery-Thoracic"="49",	"Psychiatry-Addictive"="50",	"Surgery-Cardiovascular"="51",
                                            "PhysicianNotFound"="52",	"AllergyandImmunology"="53",	"Hematology"="54",
                                            "Proctology"="55",	"Rheumatology"="56",	"Obstetrics"="57",
                                            "Dentistry"="58",	"SurgicalSpecialty"="59",	"Radiologist"="60",
                                            "Pathology"="61",	"Dermatology"="62",	"SportsMedicine"="63",
                                            "Hospitalist"="64","OutreachServices"="65","Cardiology-Pediatric"="66",
                                            "Perinatology"="67","Neurophysiology"="68","Endocrinology-Metabolism"="69",
                                            "DCPTEAM"="70","Resident"="71","Surgery-PlasticwithinHeadandNeck"="72","Speech"="73")

# remove NULL character from admissiontype, admissionsource and dischargedisposition

diabetesData <- (diabetesData[-which(diabetesData$admissiontypeid == 6),])
diabetesData <- (diabetesData[-which(diabetesData$admissionsourceid == 17),])
diabetesData <- (diabetesData[-which(diabetesData$dischargedispositionid == 18),])
diabetesData <- (diabetesData[-which(diabetesData$dischargedispositionid == 19),])
#diabetesData <- (diabetesData[-which(diabetesData$dischargedispositionid == 20),])
diabetesData <- (diabetesData[-which(diabetesData$dischargedispositionid == 11),])
#diabetesData <- (diabetesData[-which(diabetesData$dischargedispositionid == 21),])

#replace value "external Injury" in diagnoses columns to number code 1001
diabetesData$diag1[diabetesData$diag1=="External Injury"] <- 1001
diabetesData$diag2[diabetesData$diag2=="External Injury"] <- 1001
diabetesData$diag3[diabetesData$diag3=="External Injury"] <- 1001
View(diabetesData)

#let's store this version so that we can use it later for decision trees
diabData_DT <- diabetesData
write.csv(diabData_DT, "diabData_DeciTree.csv")

#A quick view of the structure of the data shows that the attributes are not of the right data type.
str(diabetesData)
diabetesData$age=factor(diabetesData$age)
diabetesData$race=factor(diabetesData$race)
diabetesData$admissiontypeid=factor(diabetesData$admissiontypeid)
str(diabetesData)

# read from the csv file
RevisedDiabData <- diabetesData
View(RevisedDiabData)
summary(RevisedDiabData)

write.csv(RevisedDiabData,"diabetic1.csv")

#check to see if there are any more NA values
Total <-sum(is.na(RevisedDiabData))
cat("The number of missing values in this data is ", Total )


#### Exploratory Data Analysis ---------------------------------------------

#visualization of data
visualDiab <- RevisedDiabData
gg1 <- ggplot(visualDiab, aes(x= visualDiab$readmitted, fill = visualDiab$age ) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by Age")
gg2 <- ggplot(visualDiab, aes(x= visualDiab$readmitted, fill = visualDiab$gender ) ) + geom_bar(position = "dodge")   + ggtitle("Readmission by Gender")
gg3 <- ggplot(visualDiab, aes(x= visualDiab$readmitted,fill = visualDiab$race ) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by Race") 
gg5 <- ggplot(visualDiab, aes(x= visualDiab$readmitted,fill = visualDiab$admissiontypeid) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by admission type") 


gg1
gg2
gg3
gg5

#numerical correlations
library(corrplot)
numeric_data<-select_if(visualDiab,is.numeric)
c <- cor(numeric_data, use= "pairwise.complete.obs")
corrplot(c)

##correlation visualization
visualDiab2 <- RevisedDiabData
str(visualDiab2)
visualDiab2$diag1 <- as.integer(visualDiab2$diag1)
visualDiab2$diag2 <- as.integer(visualDiab2$diag2)
visualDiab2$diag3 <- as.integer(visualDiab2$diag3)
visualDiab2$medicalspecialty <- as.integer(visualDiab2$medicalspecialty)
visualDiab2$race=dplyr::recode(visualDiab2$race, "Caucasian"=1, "AfricanAmerican"=2, "Asian"=3, "Hispanic"=4,"Other"=5,"Unknown"=6)
visualDiab2$gender=dplyr::recode(visualDiab2$gender, "Female"="1", "Male"="0")
visualDiab2$gender <- as.integer(visualDiab2$gender)
visualDiab2$race <- as.integer(visualDiab2$race)
visualDiab2$age=dplyr::recode(visualDiab2$age, "[0-10]"="child","[10-20]"="teens", "[20-30]"="twenties","[30-40]"="thirties","[40-50]"="forties",
                                  "[50-60]"="fifties","[60-70]"="sixties","[70-80]"="Seventies","[80-90]"="Eighties","[90-100]"="Ninties")
visualDiab2$age=dplyr::recode(visualDiab2$age, "child"=0,"teens"=1, "twenties"=2,"thirties"=3,"forties"=4,
                                 "fifties"=5,"sixties"=6,"Seventies"=7,"Eighties"=8,"Ninties"=9)
visualDiab2$age <- as.integer(visualDiab2$age)
#converting and recoding all the character values of medications
visualDiab2$maxgluserum=dplyr::recode(visualDiab2$maxgluserum, ">200"=1, ">300"=2, "None"=0,"Norm"=3)
visualDiab2$maxgluserum <- as.integer(visualDiab2$maxgluserum)
visualDiab2$a1cresult=dplyr::recode(visualDiab2$a1cresult, ">7"=1, ">8"=2, "None"=0,"Norm"=3)
visualDiab2$a1cresult<- as.integer(visualDiab2$a1cresult)
visualDiab2$metformin=dplyr::recode(visualDiab2$metformin, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$metformin<- as.integer(visualDiab2$metformin)
visualDiab2$repaglinide=dplyr::recode(visualDiab2$repaglinide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$repaglinide<- as.integer(visualDiab2$repaglinide)
visualDiab2$nateglinide=dplyr::recode(visualDiab2$nateglinide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$nateglinide<- as.integer(visualDiab2$nateglinide)
visualDiab2$chlorpropamide=dplyr::recode(visualDiab2$chlorpropamide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$chlorpropamide<- as.integer(visualDiab2$chlorpropamide)
visualDiab2$glimepiride=dplyr::recode(visualDiab2$glimepiride, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$glimepiride<- as.integer(visualDiab2$glimepiride)
visualDiab2$glipizide=dplyr::recode(visualDiab2$glipizide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$glipizide<- as.integer(visualDiab2$glipizide)
visualDiab2$glyburide=dplyr::recode(visualDiab2$glyburide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$glyburide<- as.integer(visualDiab2$glyburide)
visualDiab2$tolbutamide=dplyr::recode(visualDiab2$tolbutamide,  "No"=0, "Steady"=1)
visualDiab2$tolbutamide<- as.integer(visualDiab2$tolbutamide)
visualDiab2$pioglitazone=dplyr::recode(visualDiab2$pioglitazone, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$pioglitazone<- as.integer(visualDiab2$pioglitazone)
visualDiab2$rosiglitazone=dplyr::recode(visualDiab2$rosiglitazone, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$rosiglitazone<- as.integer(visualDiab2$rosiglitazone)
visualDiab2$acarbose=dplyr::recode(visualDiab2$acarbose, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$acarbose<- as.integer(visualDiab2$acarbose)
visualDiab2$miglitol=dplyr::recode(visualDiab2$miglitol, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$miglitol<- as.integer(visualDiab2$miglitol)
visualDiab2$tolazamide=dplyr::recode(visualDiab2$tolazamide, "No"=0, "Steady"=1, "Up"=3)
visualDiab2$tolazamide<- as.integer(visualDiab2$tolazamide)
visualDiab2$insulin=dplyr::recode(visualDiab2$insulin, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$insulin<- as.integer(visualDiab2$insulin)
visualDiab2$glyburide.metformin=dplyr::recode(visualDiab2$glyburide.metformin, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
visualDiab2$glyburide.metformin<- as.integer(visualDiab2$glyburide.metformin)
visualDiab2$glipizide.metformin=dplyr::recode(visualDiab2$glipizide.metformin, "No"=0, "Steady"=2)
visualDiab2$glipizide.metformin<- as.integer(visualDiab2$glipizide.metformin)
visualDiab2$glimepiride.pioglitazone=dplyr::recode(visualDiab2$glimepiride.pioglitazone, "No"=0, "Steady"=2)
visualDiab2$glimepiride.pioglitazone<- as.integer(visualDiab2$glimepiride.pioglitazone)
visualDiab2$metformin.rosiglitazone=dplyr::recode(visualDiab2$metformin.rosiglitazone, "No"=0, "Steady"=2)
visualDiab2$metformin.rosiglitazone<- as.integer(visualDiab2$metformin.rosiglitazone)
visualDiab2$metformin.pioglitazone=dplyr::recode(visualDiab2$metformin.pioglitazone, "No"=0, "Steady"=2)
visualDiab2$metformin.pioglitazone<- as.integer(visualDiab2$metformin.pioglitazone)
visualDiab2$change=dplyr::recode(visualDiab2$change, "No"=0, "Ch"=1)
visualDiab2$change<- as.integer(visualDiab2$change)
visualDiab2$diabetesmed=dplyr::recode(visualDiab2$diabetesmed, "No"=0, "Yes"=1)
visualDiab2$diabetesmed<- as.integer(visualDiab2$diabetesmed)


rosiglitazone<-visualDiab2$rosiglitazone
pioglitazone<-visualDiab2$pioglitazone
tolbutamide<-visualDiab2$tolbutamide
glyburide<-visualDiab2$glyburide
glipizide<-visualDiab2$glipizide
glimepiride<-visualDiab2$glimepiride
chlorpropamide<-visualDiab2$chlorpropamide
nateglinide<-visualDiab2$nateglinide
repaglinide<-visualDiab2$repaglinide
metformin<-visualDiab2$metformin
a1cresult<-visualDiab2$a1cresult
maxgluserum<-visualDiab2$maxgluserum
age<-visualDiab2$age
gender<-visualDiab2$gender
race<-visualDiab2$race
acarbose<-visualDiab2$acarbose
miglitol<-visualDiab2$miglitol
tolazamide<-visualDiab2$tolazamid
insulin<-visualDiab2$insulin
glyburide.metformin<-visualDiab2$glyburide.metformin
glipizide.metformin<-visualDiab2$glipizide.metformin
glimepiride.pioglitazone<-visualDiab2$glimepiride.pioglitazone
metformin.rosiglitazone<-visualDiab2$metformin.rosiglitazone
metformin.pioglitazone <- visualDiab2$metformin.pioglitazone
diabetesmed <- visualDiab2$diabetesmed
change <- visualDiab2$change
diagnosis1 <- visualDiab2$diag1
diagnosis2 <- visualDiab2$diag2
diagnosis3 <- visualDiab2$diag3
MedicalSpecialty <- visualDiab2$medicalspecialty


newdata <- data.frame(diabetesmed,change,diagnosis1,
                      diagnosis2,diagnosis3,MedicalSpecialty,age,race,gender,rosiglitazone,pioglitazone,tolbutamide,
                      glyburide,glipizide,glimepiride,chlorpropamide,nateglinide,repaglinide,metformin,a1cresult,
                      maxgluserum,acarbose,miglitol,tolazamide,insulin,glyburide.metformin,
                      glipizide.metformin,glimepiride.pioglitazone,metformin.rosiglitazone,metformin.pioglitazone)
View(newdata)
c1 <- cor(newdata, use= "pairwise.complete.obs")
corrplot(c1)
#the results show a strong correlation between diabetesmed, change and insulin levels. Let's remove insulin and diabetesmed for 
#further analysis.

#visualization for Readmission rates by Diagnosis
visualDiab$diag1[visualDiab$diag1 == 0] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 1 & visualDiab$diag1 <= 139] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 >= 390 & visualDiab$diag1 <= 459] <- "Circulatory"
visualDiab$diag1[visualDiab$diag1 == 785] <- "Circulatory"
visualDiab$diag1[visualDiab$diag1 >= 460 & visualDiab$diag1 <= 519] <- "Respiratory"
visualDiab$diag1[visualDiab$diag1 == 786] <- "Respiratory"
visualDiab$diag1[visualDiab$diag1 >= 520 & visualDiab$diag1 <= 579] <- "Digestive"
visualDiab$diag1[visualDiab$diag1 == 787] <- "Digestive"
visualDiab$diag1[visualDiab$diag1 == 250] <- "Diabetes"
visualDiab$diag1[visualDiab$diag1 >= 800 & visualDiab$diag1 <= 999] <- "Injury"
visualDiab$diag1[visualDiab$diag1 >= 710 & visualDiab$diag1 <= 739] <- "Musculoskeletal"
visualDiab$diag1[visualDiab$diag1 >= 580 & visualDiab$diag1 <= 629] <- "Genitourinary"
visualDiab$diag1[visualDiab$diag1 == 788] <- "Genitourinary"
visualDiab$diag1[visualDiab$diag1 >= 140 & visualDiab$diag1 <= 239] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 >= 790 & visualDiab$diag1 <= 799] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 == 780] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 == 781] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 == 782] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 == 784] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 >= 240 & visualDiab$diag1 <= 279] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 >= 680 & visualDiab$diag1 <= 709] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 >= 290 & visualDiab$diag1 <= 319] <- "Other"
visualDiab$diag1[visualDiab$diag1 == 1001] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 280 & visualDiab$diag1 <= 289] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 320 & visualDiab$diag1 <= 359] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 630 & visualDiab$diag1 <= 679] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 360 & visualDiab$diag1 <= 389] <- "Other"
visualDiab$diag1[visualDiab$diag1 >= 740 & visualDiab$diag1 <= 759] <- "Other"
visualDiab$diag1[visualDiab$diag1 == 783] <- "Other"
visualDiab$diag1[visualDiab$diag1 == 789] <- "Other"
visualDiab$diag1[visualDiab$diag1 >=36 & visualDiab$diag1 <= 79] <- "Neoplasms"
visualDiab$diag1[visualDiab$diag1 == 8] <- "Neoplasms"
gg4 <- ggplot(visualDiab, aes(x= visualDiab$readmitted,fill = visualDiab$diag1 ) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by Diagnosis") 
gg4

#---------------Removing some more variables and transforming some other after the above visualizations

RevisedDiabData$diag1[RevisedDiabData$diag1 == 0] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 1 & RevisedDiabData$diag1 <= 139] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 390 & RevisedDiabData$diag1 <= 459] <- "Circulatory"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 785] <- "Circulatory"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 460 & RevisedDiabData$diag1 <= 519] <- "Respiratory"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 786] <- "Respiratory"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 520 & RevisedDiabData$diag1 <= 579] <- "Digestive"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 787] <- "Digestive"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 250] <- "Diabetes"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 800 & RevisedDiabData$diag1 <= 999] <- "Injury"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 710 & RevisedDiabData$diag1 <= 739] <- "Musculoskeletal"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 580 & RevisedDiabData$diag1 <= 629] <- "Genitourinary"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 788] <- "Genitourinary"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 140 & RevisedDiabData$diag1 <= 239] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 790 & RevisedDiabData$diag1 <= 799] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 780] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 781] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 782] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 784] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 240 & RevisedDiabData$diag1 <= 279] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 680 & RevisedDiabData$diag1 <= 709] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 290 & RevisedDiabData$diag1 <= 319] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 1001] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 280 & RevisedDiabData$diag1 <= 289] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 320 & RevisedDiabData$diag1 <= 359] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 630 & RevisedDiabData$diag1 <= 679] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 360 & RevisedDiabData$diag1 <= 389] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >= 740 & RevisedDiabData$diag1 <= 759] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 783] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 789] <- "Other"
RevisedDiabData$diag1[RevisedDiabData$diag1 >=36 & RevisedDiabData$diag1 <= 79] <- "Neoplasms"
RevisedDiabData$diag1[RevisedDiabData$diag1 == 8] <- "Neoplasms"
RevisedDiabData$diag1 <- as.factor(RevisedDiabData$diag1)
RevisedDiabData$diag2[RevisedDiabData$diag2 == 0] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 1 & RevisedDiabData$diag2 <= 139] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 390 & RevisedDiabData$diag2 <= 459] <- "Circulatory"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 785] <- "Circulatory"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 460 & RevisedDiabData$diag2 <= 519] <- "Respiratory"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 786] <- "Respiratory"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 520 & RevisedDiabData$diag2 <= 579] <- "Digestive"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 787] <- "Digestive"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 250] <- "Diabetes"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 800 & RevisedDiabData$diag2 <= 999] <- "Injury"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 710 & RevisedDiabData$diag2 <= 739] <- "Musculoskeletal"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 580 & RevisedDiabData$diag2 <= 629] <- "Genitourinary"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 788] <- "Genitourinary"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 140 & RevisedDiabData$diag2 <= 239] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 790 & RevisedDiabData$diag2 <= 799] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 780] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 781] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 782] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 784] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 240 & RevisedDiabData$diag2 <= 279] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 680 & RevisedDiabData$diag2 <= 709] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 290 & RevisedDiabData$diag2 <= 319] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 1001] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 280 & RevisedDiabData$diag2 <= 289] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 320 & RevisedDiabData$diag2 <= 359] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 630 & RevisedDiabData$diag2 <= 679] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 360 & RevisedDiabData$diag2 <= 389] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >= 740 & RevisedDiabData$diag2 <= 759] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 783] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 789] <- "Other"
RevisedDiabData$diag2[RevisedDiabData$diag2 >=36 & RevisedDiabData$diag2 <= 79] <- "Neoplasms"
RevisedDiabData$diag2[RevisedDiabData$diag2 == 8] <- "Neoplasms"
RevisedDiabData$diag2 <- as.factor(RevisedDiabData$diag2)
RevisedDiabData$diag3[RevisedDiabData$diag3 == 0] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 1 & RevisedDiabData$diag3 <= 139] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 390 & RevisedDiabData$diag3 <= 459] <- "Circulatory"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 785] <- "Circulatory"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 460 & RevisedDiabData$diag3 <= 519] <- "Respiratory"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 786] <- "Respiratory"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 520 & RevisedDiabData$diag3 <= 579] <- "Digestive"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 787] <- "Digestive"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 250] <- "Diabetes"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 800 & RevisedDiabData$diag3 <= 999] <- "Injury"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 710 & RevisedDiabData$diag3 <= 739] <- "Musculoskeletal"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 580 & RevisedDiabData$diag3 <= 629] <- "Genitourinary"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 788] <- "Genitourinary"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 140 & RevisedDiabData$diag3 <= 239] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 790 & RevisedDiabData$diag3 <= 799] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 780] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 781] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 782] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 784] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 240 & RevisedDiabData$diag3 <= 279] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 680 & RevisedDiabData$diag3 <= 709] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 290 & RevisedDiabData$diag3 <= 319] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 1001] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 280 & RevisedDiabData$diag3 <= 289] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 320 & RevisedDiabData$diag3 <= 359] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 630 & RevisedDiabData$diag3 <= 679] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 360 & RevisedDiabData$diag3 <= 389] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >= 740 & RevisedDiabData$diag3 <= 759] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 783] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 789] <- "Other"
RevisedDiabData$diag3[RevisedDiabData$diag3 >=36 & RevisedDiabData$diag3 <= 79] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 8] <- "Neoplasms"
RevisedDiabData$diag3[RevisedDiabData$diag3 == 14] <- "Neoplasms"
RevisedDiabData$diag3 <- as.factor(RevisedDiabData$diag3)

RevisedDiabData$age=dplyr::recode(RevisedDiabData$age, "[0-10]"="child","[10-20]"="teens", "[20-30]"="twenties","[30-40]"="thirties","[40-50]"="forties",
                                      "[50-60]"="fifties","[60-70]"="sixties","[70-80]"="Seventies","[80-90]"="Eighties","[90-100]"="Ninties")


#Let's discretize medical specialty into small groups
## Place into eight groups: 1 - Unknown, 2-Obstetrics&Gynaec 3-Psychology 4-Sports & Ortho 5-family medicine and others
# 6- Pediatrics 7-Surgery 8-All other fields of medicine
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 52] <- 1
#RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 1] <- 1
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 2] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 3] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 4] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 5] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 6] <- 4
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 7] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 13] <- 2
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 57] <- 2
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 19] <- 2
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 31] <- 2
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 25] <- 3
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 28] <- 3
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 11] <- 3
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 50] <- 3
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 63] <- 4
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 37] <- 4

RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 9] <- 4
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 71] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 65] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 64] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 70] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 60] <- 5
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 66] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 14] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 20] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 23] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 32] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 35] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 18] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 21] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 30] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 36] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 44] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 45] <- 6
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 22] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 43] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 46] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 49] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 29] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 47] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 59] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 72] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 33] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 51] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 12] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 24] <- 7
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 27] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 42] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 53] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 56] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 62] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 39] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 48] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 54] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 73] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 10] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 16] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 34] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 40] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 55] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 58] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 61] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 67] <- 8
#RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 8] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 17] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 38] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 41] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 68] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 26] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 15] <- 8
RevisedDiabData$medicalspecialty[RevisedDiabData$medicalspecialty == 69] <- 8

RevisedDiabData$medicalspecialty <- as.factor(RevisedDiabData$medicalspecialty)

#discretizing numprocedures
RevisedDiabData$numprocedures[RevisedDiabData$numprocedures <=3] <- 1
RevisedDiabData$numprocedures[RevisedDiabData$numprocedures >3] <- 2
RevisedDiabData$numprocedures <- as.factor(RevisedDiabData$numprocedures)

#discretizing numlabprocedures
RevisedDiabData$numlabprocedures[RevisedDiabData$numlabprocedures <=44] <- 1
RevisedDiabData$numlabprocedures[RevisedDiabData$numlabprocedures >44] <- 2
RevisedDiabData$numlabprocedures <- as.factor(RevisedDiabData$numlabprocedures)

#discretizing timein hospital
RevisedDiabData$timeinhospital[RevisedDiabData$timeinhospital <=5] <- 1
RevisedDiabData$timeinhospital[RevisedDiabData$timeinhospital >=6 &  RevisedDiabData$timeinhospital<=10] <- 2
RevisedDiabData$timeinhospital[RevisedDiabData$timeinhospital >=11 & RevisedDiabData$timeinhospital <=14] <- 3
RevisedDiabData$timeinhospital <- as.factor(RevisedDiabData$timeinhospital)


#Another column that is important is discharge_disposition_id, which tells us where the patient went
#after the hospitalization. If we look at the IDs_mapping.csv provided by UCI we can see that 11,13,14,19,20,21 
#are related to death or hospice. We should remove these samples from the predictive model since they cannot be readmitted.
#Let's discretize dichargedispositionid into small groups
## Place into four groups: 1: Discharged to Home, 2: Discharged to Another Facility\Hospital, 3: Admitted,  4: Unknown
RevisedDiabData$dischargedispositionid <- as.integer(RevisedDiabData$dischargedispositionid)
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 1] <- 1
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 6] <- 1
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 8] <- 1
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==13] <- 1
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==3] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==4] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==5] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==10] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==14] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==16] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==22] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==23] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==24] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==27] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==28] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==29] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid ==30] <- 2
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 9] <- 3
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 12] <- 3
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 15] <- 3
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 17] <- 3
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 25] <- 4
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 26] <- 4
RevisedDiabData$dischargedispositionid[RevisedDiabData$dischargedispositionid == 7] <- 4
RevisedDiabData$dischargedispositionid <- as.factor(RevisedDiabData$dischargedispositionid)

#admissionsourceid:- 1-Referrals, 2-Transfers, 3-Admission from a diff dept, 4-Unknown
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid <= 3] <- 1
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid == 10] <- 2
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid == 18] <- 2
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid >= 4 & RevisedDiabData$admissionsourceid <=6] <- 2
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid >= 24 & RevisedDiabData$admissionsourceid <=26] <- 2
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid >= 7 & RevisedDiabData$admissionsourceid <=8] <- 3
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid >= 11 & RevisedDiabData$admissionsourceid <=14] <- 3
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==19] <- 3
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==22] <- 3
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==23] <- 3
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==9] <- 4
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==15] <- 4
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==20] <- 4
RevisedDiabData$admissionsourceid[RevisedDiabData$admissionsourceid ==21] <- 4
RevisedDiabData$admissionsourceid <- as.factor(RevisedDiabData$admissionsourceid)

RevisedDiabData$readmitted <- as.factor(RevisedDiabData$readmitted)
str(RevisedDiabData)

#below columns have only one level or one type of value. Hence, they don't contribute and are removed.
RevisedDiabData$examide <- NULL
RevisedDiabData$citoglipton <- NULL
RevisedDiabData$troglitazone <- NULL
RevisedDiabData$acetohexamide <- NULL

#remove nummedications since it has a correlation with timein hospital
RevisedDiabData$nummedications <- NULL
RevisedDiabData$change <- NULL
RevisedDiabData$insulin <- NULL
write.csv(RevisedDiabData, "reviseddiabdata.csv")

#####More EDA----------------
gg6 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$dischargedispositionid) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by discharge disposition") 
gg7 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$admissionsourceid) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by admission source") 
gg8 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numberoutpatient) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of outpatient visits") 
gg9 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numberinpatient) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of inpatient visits") 
gg10 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numberemergency) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of emergency visits") 
gg11 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numberdiagnoses) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of diagnoses") 
gg13 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$a1cresult) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by a1c tests") 
gg14 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$diabetesmed) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by diabetes meds") 
gg15 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numlabprocedures) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of lab procedures") 
gg16 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$numprocedures) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by number of procedures") 
gg18 <- ggplot(RevisedDiabData, aes(x= RevisedDiabData$readmitted,fill = RevisedDiabData$timeinhospital) ) + geom_bar(position = "dodge")  + ggtitle("Readmission by time in hospital") 

gg6
gg7
gg8
gg9
gg10
gg11
gg13
gg14
gg15
gg16
gg18


#---------------ARM Technique----------------------------------------------------
#To run the apriori algorithm, all variables should be of factor datatype. 
#So, let's first create a new data variable since this only applies to the ARM algorithm
DiabData_ARM <- RevisedDiabData
str(DiabData_ARM)

DiabData_ARM$numberoutpatient <- as.factor(DiabData_ARM$numberoutpatient)
DiabData_ARM$numberemergency <- as.factor(DiabData_ARM$numberemergency)
DiabData_ARM$numberinpatient <- as.factor(DiabData_ARM$numberinpatient)
DiabData_ARM$numberdiagnoses <- as.factor(DiabData_ARM$numberdiagnoses)

## Now load the transformed data into the apriori algorithm 
library(arules)
library(arulesViz)
myRules = apriori(DiabData_ARM, parameter=list(supp=0.001,conf = 0.08, minlen=2), 
                  appearance = list(default="lhs",rhs="readmitted=1"),
                  control = list(verbose=F))
summary(myRules)

# Show the top 5 rules, rounding with 2 digits
options(digits=2)
arules::inspect(myRules[1:5])

# Sort rules so that we can view the most relevant rules first. For example, sort rules with "confidence":
myRules_relevant<-sort(myRules, by="confidence", decreasing=TRUE)
arules::inspect(myRules_relevant[1:20])
#There are some pretty interesting rules.

# Visualize the rules
plot(myRules_relevant, measure = c("support", "lift"), shading = "confidence", jitter=0)
plot(myRules_relevant[1:20], method="graph")

#ARM rules when readmitted=0
myRules2 = apriori(DiabData_ARM, parameter=list(supp=0.001,conf = 0.08, minlen=2), 
                  appearance = list(default="lhs",rhs="readmitted=0"),
                  control = list(verbose=F))
summary(myRules2)
# Show the top 5 rules, rounding with 2 digits
options(digits=2)
arules::inspect(myRules2[1:5])
# Sort rules so that we can view the most relevant rules first. For example, sort rules with "confidence":
myRules2_relevant<-sort(myRules2, by="confidence", decreasing=TRUE)
arules::inspect(myRules2_relevant[1:20])
# Visualize the rules
plot(myRules2_relevant, measure = c("support", "lift"), shading = "confidence", jitter=0)
plot(myRules2_relevant[1:20], method="graph")

#ARM rules when readmitted=2
myRules3 = apriori(DiabData_ARM, parameter=list(supp=0.001,conf = 0.08, minlen=2), 
                   appearance = list(default="lhs",rhs="readmitted=2"),
                   control = list(verbose=F))
summary(myRules3)
# Show the top 5 rules, rounding with 2 digits
options(digits=2)
arules::inspect(myRules3[1:5])
# Sort rules so that we can view the most relevant rules first. For example, sort rules with "confidence":
myRules3_relevant<-sort(myRules3, by="confidence", decreasing=TRUE)
arules::inspect(myRules3_relevant[1:20])
# Visualize the rules
plot(myRules3_relevant, measure = c("support", "lift"), shading = "confidence", jitter=0)
plot(myRules3_relevant[1:20], method="graph")


#------------------------------###CLUSTERING-----------------------
#For clustering, we can't use the same preprocessed dataset as the ARM since all variables were converted to factor type
#So, need to do different pre-processing on the RevisedDiabData dataset. 
#install.packages("cluster")
library(cluster)
DiabData_Clust <- RevisedDiabData
str(DiabData_Clust)
DiabData_Clust$race=dplyr::recode(DiabData_Clust$race, "Caucasian"=1, "AfricanAmerican"=2, "Asian"=3, "Hispanic"=4,"Other"=5,"Unknown"=6)
DiabData_Clust$gender=dplyr::recode(DiabData_Clust$gender, "Female"="1", "Male"="0")
DiabData_Clust$gender <- as.integer(DiabData_Clust$gender)
DiabData_Clust$race <- as.integer(DiabData_Clust$race)

DiabData_Clust$age=dplyr::recode(DiabData_Clust$age, "child"=0,"teens"=1, "twenties"=2,"thirties"=3,"forties"=4,
                                      "fifties"=5,"sixties"=6,"Seventies"=7,"Eighties"=8,"Ninties"=9)
DiabData_Clust$age <- as.integer(DiabData_Clust$age)
DiabData_Clust$medicalspecialty <- as.integer(DiabData_Clust$medicalspecialty)
DiabData_Clust$diag1 <- as.integer(DiabData_Clust$diag1)
DiabData_Clust$diag2 <- as.integer(DiabData_Clust$diag2)
DiabData_Clust$diag3 <- as.integer(DiabData_Clust$diag3)
DiabData_Clust$admissiontypeid <- as.integer(DiabData_Clust$admissiontypeid)
DiabData_Clust$admissionsourceid <- as.integer(DiabData_Clust$admissionsourceid)
DiabData_Clust$timeinhospital <- as.integer(DiabData_Clust$timeinhospital)
DiabData_Clust$dischargedispositionid <- as.integer(DiabData_Clust$dischargedispositionid)
DiabData_Clust$numlabprocedures <- as.integer(DiabData_Clust$numlabprocedures)
DiabData_Clust$numprocedures <- as.integer(DiabData_Clust$numprocedures)

#converting and recoding all the character values of medications
DiabData_Clust$maxgluserum=dplyr::recode(DiabData_Clust$maxgluserum, ">200"=1, ">300"=2, "None"=0,"Norm"=3)
DiabData_Clust$maxgluserum <- as.integer(DiabData_Clust$maxgluserum)
DiabData_Clust$a1cresult=dplyr::recode(DiabData_Clust$a1cresult, ">7"=1, ">8"=2, "None"=0,"Norm"=3)
DiabData_Clust$a1cresult<- as.integer(DiabData_Clust$a1cresult)
DiabData_Clust$metformin=dplyr::recode(DiabData_Clust$metformin, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$metformin<- as.integer(DiabData_Clust$metformin)
DiabData_Clust$repaglinide=dplyr::recode(DiabData_Clust$repaglinide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$repaglinide<- as.integer(DiabData_Clust$repaglinide)
DiabData_Clust$nateglinide=dplyr::recode(DiabData_Clust$nateglinide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$nateglinide<- as.integer(DiabData_Clust$nateglinide)
DiabData_Clust$chlorpropamide=dplyr::recode(DiabData_Clust$chlorpropamide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$chlorpropamide<- as.integer(DiabData_Clust$chlorpropamide)
DiabData_Clust$glimepiride=dplyr::recode(DiabData_Clust$glimepiride, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$glimepiride<- as.integer(DiabData_Clust$glimepiride)
DiabData_Clust$glipizide=dplyr::recode(DiabData_Clust$glipizide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$glipizide<- as.integer(DiabData_Clust$glipizide)
DiabData_Clust$glyburide=dplyr::recode(DiabData_Clust$glyburide, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$glyburide<- as.integer(DiabData_Clust$glyburide)
DiabData_Clust$tolbutamide=dplyr::recode(DiabData_Clust$tolbutamide,  "No"=0, "Steady"=1)
DiabData_Clust$tolbutamide<- as.integer(DiabData_Clust$tolbutamide)
DiabData_Clust$pioglitazone=dplyr::recode(DiabData_Clust$pioglitazone, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$pioglitazone<- as.integer(DiabData_Clust$pioglitazone)
DiabData_Clust$rosiglitazone=dplyr::recode(DiabData_Clust$rosiglitazone, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$rosiglitazone<- as.integer(DiabData_Clust$rosiglitazone)
DiabData_Clust$acarbose=dplyr::recode(DiabData_Clust$acarbose, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$acarbose<- as.integer(DiabData_Clust$acarbose)
DiabData_Clust$miglitol=dplyr::recode(DiabData_Clust$miglitol, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$miglitol<- as.integer(DiabData_Clust$miglitol)
DiabData_Clust$tolazamide=dplyr::recode(DiabData_Clust$tolazamide, "No"=0, "Steady"=1, "Up"=3)
DiabData_Clust$tolazamide<- as.integer(DiabData_Clust$tolazamide)
DiabData_Clust$diabetesmed=dplyr::recode(DiabData_Clust$diabetesmed, "Yes"=1,"No"=0)
DiabData_Clust$diabetesmed<- as.integer(DiabData_Clust$diabetesmed)
DiabData_Clust$glyburide.metformin=dplyr::recode(DiabData_Clust$glyburide.metformin, "Down"=1, "No"=0, "Steady"=2,"Up"=3)
DiabData_Clust$glyburide.metformin<- as.integer(DiabData_Clust$glyburide.metformin)
DiabData_Clust$glipizide.metformin=dplyr::recode(DiabData_Clust$glipizide.metformin, "No"=0, "Steady"=2)
DiabData_Clust$glipizide.metformin<- as.integer(DiabData_Clust$glipizide.metformin)
DiabData_Clust$glimepiride.pioglitazone=dplyr::recode(DiabData_Clust$glimepiride.pioglitazone, "No"=0, "Steady"=2)
DiabData_Clust$glimepiride.pioglitazone<- as.integer(DiabData_Clust$glimepiride.pioglitazone)
DiabData_Clust$metformin.rosiglitazone=dplyr::recode(DiabData_Clust$metformin.rosiglitazone, "No"=0, "Steady"=2)
DiabData_Clust$metformin.rosiglitazone<- as.integer(DiabData_Clust$metformin.rosiglitazone)
DiabData_Clust$metformin.pioglitazone=dplyr::recode(DiabData_Clust$metformin.pioglitazone, "No"=0, "Steady"=2)
DiabData_Clust$metformin.pioglitazone<- as.integer(DiabData_Clust$metformin.pioglitazone)

str(DiabData_Clust)
write.csv(DiabData_Clust, "diabetesDataCluster.csv")
Total <-sum(is.na(DiabData_Clust))
cat("The number of missing values in this data is ", Total )

View(DiabData_Clust)

#NORMALIZE
### Create a function to use min-max to re-scale/normalize the numerical attributes
Min_Max_function <- function(x){
  return(  (x - min(x)) /(max(x) - min(x))   )
}

## Next, apply the Min_Max to all the data.
Norm_Diab_Cluster <- as.data.frame(lapply(DiabData_Clust[,c(1:38)], Min_Max_function))
Diab_Cluster_Readmitted <- DiabData_Clust[,39]


Total <-sum(is.na(Norm_Diab_Cluster))
cat("The number of missing values in this data is ", Total )
write.csv(Norm_Diab_Cluster, "diabetesDataCluster_Norm.csv")

## Now, let's add back the labels
(head(Norm_Diab <- data.frame(Norm_Diab_Cluster, Readmitted=Diab_Cluster_Readmitted)))

#saving the normalized integer-based dataset to use for decision tree analysis later.
diabData_DT_norm <- Norm_Diab

#elbow plot to select K
elbow_plot <- function(Norm_Diab_Cluster, nc=15, seed=1234){
  wss <- (nrow(Norm_Diab_Cluster)-1)*sum(apply(Norm_Diab_Cluster,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(Norm_Diab_Cluster, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}


elbow_plot(Norm_Diab_Cluster, nc=8)

#clustering model with k=4
#let's take the dataset that doesn't have the labels but is normalized.
kmeans_Norm_Diab <- kmeans(Norm_Diab_Cluster, 4)
print(kmeans_Norm_Diab)
#the result shows 4 clusters
summary(kmeans_Norm_Diab)
Clust_K <- (table(kmeans_Norm_Diab$cluster,Diab_Cluster_Readmitted))
plot(Clust_K)
#install.packages("factoextra")
library(factoextra)
fviz_cluster(object=kmeans_Norm_Diab,data=Norm_Diab_Cluster,
             stand=FALSE,
             ellipse.type = "norm")
#with two specific variables
fviz_cluster(object=kmeans_Norm_Diab,data=Norm_Diab_Cluster,choose.vars = c("medicalspecialty","admissionsourceid"),
             stand=FALSE,
             ellipse.type = "norm")


#clustering model with k=5
#let's take the dataset that doesn't have the labels but is normalized.
kmeans_Norm_Diab2 <- kmeans(Norm_Diab_Cluster, 5)
print(kmeans_Norm_Diab2)
#the result shows 5 clusters
summary(kmeans_Norm_Diab2)
Clust_K2 <- (table(kmeans_Norm_Diab2$cluster,Diab_Cluster_Readmitted))
plot(Clust_K2)
fviz_cluster(object=kmeans_Norm_Diab2,data=Norm_Diab_Cluster,
             stand=FALSE,
             ellipse.type = "norm")
#with two specific variables
fviz_cluster(object=kmeans_Norm_Diab2,data=Norm_Diab_Cluster,choose.vars = c("medicalspecialty","admissionsourceid"),
             stand=FALSE,
             ellipse.type = "norm")


#clustering model with k=3
#let's take the dataset that doesn't have the labels but is normalized.
kmeans_Norm_Diab3 <- kmeans(Norm_Diab_Cluster, 3)
print(kmeans_Norm_Diab3)
#the result shows 5 clusters
summary(kmeans_Norm_Diab3)
Clust_K3 <- (table(kmeans_Norm_Diab3$cluster,Diab_Cluster_Readmitted))
plot(Clust_K3)
fviz_cluster(object=kmeans_Norm_Diab3,data=Norm_Diab_Cluster,
             stand=FALSE,
             ellipse.type = "norm")
#with two specific variables
fviz_cluster(object=kmeans_Norm_Diab3,data=Norm_Diab_Cluster,choose.vars = c("medicalspecialty","admissionsourceid"),
             stand=FALSE,
             ellipse.type = "norm")

#--------- DECISION TREE-------------

library(rattle)
library(rpart)
library(rpart.plot)

#-------------------pre-processing for Decision Tree--------------------------
#Changing variables into categorical data
diabData_DT_cat <- RevisedDiabData
str(diabData_DT_cat)
diabData_DT_cat$medicalspecialty <- as.integer(diabData_DT_cat$medicalspecialty)
#medical specialty :- 1 - Unknown, 2-Obstetrics&Gynaec 3-Psychology 4-Sports & Ortho 5-family medicine and others
#6- Pediatrics 7-Surgery 8-All other fields of medicine
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 1] <- "Unknown"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 2] <- "Obstet&Gynaec"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 3] <- "Psychology"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 4] <- "Sports & Ortho"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 5] <- "Family Med & others"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 6] <- "Pediatrics"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 7] <- "Surgery"
diabData_DT_cat$medicalspecialty[diabData_DT_cat$medicalspecialty == 8] <- "AllOtherFields"
diabData_DT_cat$medicalspecialty <- as.factor(diabData_DT_cat$medicalspecialty)
View(diabData_DT_cat)

#numprocedures :- 1 - less than 3, 2- greater than 3
diabData_DT_cat$numprocedures <- as.integer(diabData_DT_cat$numprocedures)
diabData_DT_cat$numprocedures[diabData_DT_cat$numprocedures == 1] <- "LessThan3"
diabData_DT_cat$numprocedures[diabData_DT_cat$numprocedures == 2] <- "GreaterThan3"
diabData_DT_cat$numprocedures <- as.factor(diabData_DT_cat$numprocedures)
#numlabprocedures:- 1-less than 44, 2-greater than 44
diabData_DT_cat$numlabprocedures <- as.integer(diabData_DT_cat$numlabprocedures)
diabData_DT_cat$numlabprocedures[diabData_DT_cat$numlabprocedures == 1] <- "LessThan44"
diabData_DT_cat$numlabprocedures[diabData_DT_cat$numlabprocedures == 2] <- "GreaterThan44"
diabData_DT_cat$numlabprocedures <- as.factor(diabData_DT_cat$numlabprocedures)
#time in hospital:- 1- 0 to 5 days, 2- 6 to 10 days, 3 - 11 to 14 days
diabData_DT_cat$timeinhospital <- as.integer(diabData_DT_cat$timeinhospital)
diabData_DT_cat$timeinhospital[diabData_DT_cat$timeinhospital == 1] <- "Upto5Days"
diabData_DT_cat$timeinhospital[diabData_DT_cat$timeinhospital == 2] <- "6to10Days"
diabData_DT_cat$timeinhospital[diabData_DT_cat$timeinhospital == 3] <- "11to14Days"
diabData_DT_cat$timeinhospital <- as.factor(diabData_DT_cat$timeinhospital)
#dischargedispositionid:- 1: Discharged to Home, 2: Discharged to Another Facility\Hospital, 3: Admitted,  4: Unknown
diabData_DT_cat$dischargedispositionid <- as.integer(diabData_DT_cat$dischargedispositionid)
diabData_DT_cat$dischargedispositionid[diabData_DT_cat$dischargedispositionid == 1] <- "DisctoHome"
diabData_DT_cat$dischargedispositionid[diabData_DT_cat$dischargedispositionid == 2] <- "DisctoAnotherHosp"
diabData_DT_cat$dischargedispositionid[diabData_DT_cat$dischargedispositionid == 3] <- "Admitted"
diabData_DT_cat$dischargedispositionid[diabData_DT_cat$dischargedispositionid == 4] <- "Unknown"
diabData_DT_cat$dischargedispositionid <- as.factor(diabData_DT_cat$dischargedispositionid)
#admissionsourceid:- 1-Referrals, 2-Transfers, 3-Admission from a diff dept, 4-Unknown
diabData_DT_cat$admissionsourceid <- as.integer(diabData_DT_cat$admissionsourceid)
diabData_DT_cat$admissionsourceid[diabData_DT_cat$admissionsourceid == 1] <- "Referrals"
diabData_DT_cat$admissionsourceid[diabData_DT_cat$admissionsourceid == 2] <- "Transfers"
diabData_DT_cat$admissionsourceid[diabData_DT_cat$admissionsourceid == 3] <- "AdmissionfromDiffDept"
diabData_DT_cat$admissionsourceid[diabData_DT_cat$admissionsourceid == 4] <- "Unknown"
diabData_DT_cat$admissionsourceid <- as.factor(diabData_DT_cat$admissionsourceid)
#admissiontypeid:- 1-Emergency, 2-Urgent, 3-Elective, 4-Newborn, 5-Not Available, 7-Trauma Center, 8-Not Mapped
diabData_DT_cat$admissiontypeid <- as.integer(diabData_DT_cat$admissiontypeid)
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 1] <- "Emergency"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 2] <- "Urgent"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 3] <- "Elective"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 4] <- "Newborn"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 5] <- "Not Available"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 6] <- "Trauma Center"
diabData_DT_cat$admissiontypeid[diabData_DT_cat$admissiontypeid == 7] <- "Not Mapped"
diabData_DT_cat$admissiontypeid <- as.factor(diabData_DT_cat$admissiontypeid)
#numberoutpatient, emergency and inpatients discretized into categorical variables
diabData_DT_cat$numberoutpatient <- cut(diabData_DT_cat$numberoutpatient, breaks=3, 
                                        labels=c("Small","Medium", "Large"))
diabData_DT_cat$numberemergency <- cut(diabData_DT_cat$numberemergency, breaks=3, 
                                       labels=c("Small","Medium", "Large"))
diabData_DT_cat$numberinpatient <- cut(diabData_DT_cat$numberinpatient, breaks=3, 
                                       labels=c("Small","Medium", "Large"))
diabData_DT_cat$numberdiagnoses <- cut(diabData_DT_cat$numberdiagnoses, breaks=3, 
                                       labels=c("Small","Medium", "Large"))
diabData_DT_cat$readmitted <- as.integer(diabData_DT_cat$readmitted)
diabData_DT_cat$readmitted[diabData_DT_cat$readmitted == 3] <- "Not Readmitted"
diabData_DT_cat$readmitted[diabData_DT_cat$readmitted == 1] <- "< 30 Days"
diabData_DT_cat$readmitted[diabData_DT_cat$readmitted == 2] <- "Not Readmitted"
diabData_DT_cat$readmitted <- as.factor(diabData_DT_cat$readmitted)
str(diabData_DT_cat)
View(diabData_DT_cat)
#-------------------pre-processing for Decision Tree--------------------------

#this is the normalized dataset for decision tree
diabData_DT_norm
str(diabData_DT_norm)
View(diabData_DT_norm)
diabData_DT_norm$Readmitted <- as.integer(diabData_DT_norm$Readmitted)
diabData_DT_norm$Readmitted[diabData_DT_norm$Readmitted == 3] <- "Not Readmitted"
diabData_DT_norm$Readmitted[diabData_DT_norm$Readmitted == 1] <- "< 30 Days"
diabData_DT_norm$Readmitted[diabData_DT_norm$Readmitted == 2] <- "Not Readmitted"
diabData_DT_norm$Readmitted <- as.factor(diabData_DT_norm$Readmitted)
#">30"=2, "<30"=1, NO=0)
#this is the categorical dataset for decision tree
diabData_DT_cat

#let's divide the normalized dataset into training and test set by using sampling.
(n <- round(nrow(diabData_DT_norm)/5))
(s <- sample(1:nrow(diabData_DT_norm), n))
diabData_DT_Test_Norm <- diabData_DT_norm[s,]
diabData_DT_Train_Norm <- diabData_DT_norm[-s,]
#let's check how many rows are in the train and test dataset.
nrow(diabData_DT_Test_Norm)
nrow(diabData_DT_Train_Norm)
str(diabData_DT_Train_Norm)

#remove labels off the test set
diabData_DT_Test_Norm_num<- diabData_DT_Test_Norm[,c(1:38)]
Test_labels_norm <- diabData_DT_Test_Norm[,39]

write.csv(diabData_DT_Train_Norm, "diabData_DT_Train_Norm.csv")

## Create the decision tree(normalized dataset) using rpart where Y variable is the readmission column and 
#rest of the columns are X values
fit_norm <- rpart(diabData_DT_Train_Norm$Readmitted ~., data = diabData_DT_Train_Norm, method="class",
             minsplit = 1,minbucket=1, cp = 0.001)
summary(fit_norm)
plot(fit_norm, branch = 0.4,uniform = TRUE, compress = TRUE)
text(fit_norm, pretty = 0)
#predict the test dataset using the model created
predicted_norm= predict(fit_norm, diabData_DT_Test_Norm_num, type="class")
#plot the decision tree
fancyRpartPlot(fit_norm)
#confusion matrix to find correct and incorrect predictions
predmatrix <- table(Readmittance=predicted_norm, true=Test_labels_norm)
library(caret)
confusionMatrix(predmatrix)
#Accuracy is 61%, kappa is 0.22 when readmitted is recoded with 2 factors


#let's divide the categorical dataset into training and test set by using sampling.
(n <- round(nrow(diabData_DT_cat)/5))
(s <- sample(1:nrow(diabData_DT_cat), n))
diabData_DT_Test_Cat <- diabData_DT_cat[s,]
diabData_DT_Train_Cat <- diabData_DT_cat[-s,]
#let's check how many rows are in the train and test dataset.
nrow(diabData_DT_Test_Cat)
nrow(diabData_DT_Train_Cat)
str(diabData_DT_Train_Cat)
write.csv(diabData_DT_Train_Cat, "diabData_DT_Train_Cat.csv")

#remove labels off the test set
diabData_DT_Test_Cat_num<- diabData_DT_Test_Cat[,c(1:38)]
Test_labels_cat <- diabData_DT_Test_Cat[,39]

## Create the decision tree(categorical dataset) using rpart where Y variable is the readmission column and 
#rest of the columns are X values
fit_cat <- rpart(diabData_DT_Train_Cat$readmitted ~., data = diabData_DT_Train_Cat, method="class",
                  minsplit = 1,minbucket=1, cp = 0.001)
summary(fit_cat)
plot(fit_cat, branch = 0.4,uniform = TRUE, compress = TRUE)
text(fit_cat, pretty = 0)
#predict the test dataset using the model created
predicted_cat= predict(fit_cat, diabData_DT_Test_Cat_num, type="class")
#plot the decision tree
fancyRpartPlot(fit_cat)
#confusion matrix to find correct and incorrect predictions
predmatrix <- table(Readmittance=predicted_cat, true=Test_labels_cat)
confusionMatrix(predmatrix)
#Accuracy is 55%, kappa is 0.10



#trying a different tree on the normalized dataset with fewer variables to see if it improves accuracy

fit2 <- rpart(diabData_DT_Train_Norm$Readmitted ~ numberinpatient+age+a1cresult+admissionsourceid+dischargedispositionid
              +admissiontypeid+medicalspecialty+diabetesmed,
              data = diabData_DT_Train_Norm, method="class", control=rpart.control(minsplit=2,minbucket=3, cp=0.01))

plot(fit2, branch = 0.4,uniform = TRUE, compress = TRUE)
text(fit2, pretty = 0)
#plot the decision tree
fancyRpartPlot(fit2)
#predictions on test set
predicted2= predict(fit2, diabData_DT_Test_Norm_num, type="class")
predmatrix2 <- table(Readmittance=predicted2, true=Test_labels_norm)
confusionMatrix(predmatrix2)
#accuracy is 61%, kappa of 0.22 --> normalized data

#After pruning
fit2_bin = prune(fit2, cp = 0.0001)
par(mar=c(.5,.5,.5,.5))
plot(fit2_bin, branch = 0.4,uniform = TRUE, compress = TRUE)
text(fit2_bin, pretty=0)
#plot the decision tree
fancyRpartPlot(fit2_bin)
#predicting on test set
predicted3 <- predict(fit2_bin, newdata = diabData_DT_Test_Norm_num,type = "class")
predmatrix3 <- table(Readmittance=predicted3, true=Test_labels_norm)
confusionMatrix(predmatrix3)
#accuracy is still 61%, kappa 0.22 --> normalized data

diabData_DT_Train_Norm$glimepiride.pioglitazone <- NULL
diabData_DT_Train_Norm$metformin.pioglitazone <- NULL

library(CORElearn)
Method.CORElearn <- CORElearn::attrEval(diabData_DT_Train_Norm$Readmitted ~ ., data=diabData_DT_Train_Norm,  estimator = "InfGain")
(Method.CORElearn)

Method.CORElearn3 <- CORElearn::attrEval(diabData_DT_Train_Norm$Readmitted ~ ., data=diabData_DT_Train_Norm,  estimator = "Gini")
(Method.CORElearn3) 

Method.CORElearn2 <- CORElearn::attrEval(diabData_DT_Train_Norm$Readmitted ~ ., data=diabData_DT_Train_Norm,  estimator = "GainRatio")
(Method.CORElearn2)


##Naive Bayes-------------------------------------------------------------------------------
library(e1071)
library(naivebayes)
#let's divide the dataset into training and test set by using sampling.
diabDataNB <- diabData_DT_norm
(n <- round(nrow(diabDataNB)/5))
(s <- sample(1:nrow(diabDataNB), n))
diabDataNB_Test <- diabDataNB[s,]
diabDataNB_Train <- diabDataNB[-s,]
#let's check how many rows are in the train and test dataset.
nrow(diabDataNB_Test)
nrow(diabDataNB_Train)
str(diabDataNB_Train)

#remove labels off the test set
diabDataNB_Test_num<- diabDataNB_Test[,c(1:38)]
Test_NB_labels<- diabDataNB_Test[,39]
write.csv(diabDataNB, "diabData_NB.csv")


#training NB model with all variables
NB <- naiveBayes(Readmitted~., data=diabDataNB_Train, na.action = na.pass)
NB_pred <- predict(NB, diabDataNB_Test_num)
summary(NB)
NB_table <- table(NB_pred, Test_NB_labels)
confusionMatrix(NB_pred, Test_NB_labels)
plot(NB_table)
#accuracy is 60%, kappa is 0.19
library(ggplot2)
qplot(Readmitted, NB_pred, data=diabDataNB_Test,  colour=Readmitted, geom = c("boxplot", "jitter"),
                                                               main = "predicted vs. observed in validation data",
                                                               xlab = "Observed Readmissions", ylab = "Predicted Readmissions")


#let's train a different NB model with only some of the variables
#using InfoGain
library(RWeka)
Gain <- InfoGainAttributeEval(Readmitted~., data = diabDataNB_Train)
View(Gain)
# Sort Gain in descending order
Gain <- sort(Gain, decreasing = T)
barplot(Gain[1:10], col = "steel blue", las = 2)

NB2<-naiveBayes(Readmitted~numberinpatient, data=diabDataNB_Train, na.action = na.pass)
NB_pred2 <- predict(NB2, diabDataNB_Test_num)
summary(NB2)

table(NB_pred2,Test_NB_labels)
confusionMatrix(NB_pred2, Test_NB_labels)
plot(NB_pred2)
#same accuracy with 57%, kappa is 0.14


library(klaR)
diabDataNB_Train$glimepiride.pioglitazone <- NULL
diabDataNB_Train$metformin.rosiglitazone <- NULL
diabDataNB_Train$metformin.pioglitazone <- NULL
diabDataNB_Test_num$glimepiride.pioglitazone <- NULL
diabDataNB_Test_num$metformin.rosiglitazone <- NULL
diabDataNB_Test_num$metformin.pioglitazone <- NULL
View(diabDataNB_Test_num)
NB <- NaiveBayes(Readmitted~., data=diabDataNB_Train, na.action = na.pass)
NB_pred <- predict(NB, diabDataNB_Test_num)
summary(NB)
NB_table <- table(NB_pred$class, Test_NB_labels)
confusionMatrix(NB_pred$class, Test_NB_labels)
plot(NB_table)
#55% accuracy

plot(NB,n=10000,legendplot = TRUE,ylab = "Density", main = "Naive Bayes Plot")



###--------------------------- SVM------------------------------------------------------------------------

#let's divide the dataset into training and test set by using sampling.
diabDataSVM <- diabData_DT_norm
(n <- round(nrow(diabDataSVM)/5))
(s <- sample(1:nrow(diabDataSVM), n))
SVM_Test <- diabDataSVM[s,]
SVM_Train <- diabDataSVM[-s,]
#let's check how many rows are in the train and test dataset.
nrow(SVM_Test)
nrow(SVM_Train)
str(SVM_Train)

#remove labels off the test set
SVM_Test_num<- SVM_Test[,c(1:38)]
Test_SVM_labels<- SVM_Test[,39]
write.csv(diabDataSVM, "diabData_SVM.csv")


## Polynomial Kernel...
SVM_fit_P <- svm(Readmitted~., data=SVM_Train, kernel="polynomial", cost=.1, scale=FALSE)
print(SVM_fit_P)
##Prediction --
(pred_P <- predict(SVM_fit_P, SVM_Test_num, type="class"))
## COnfusion Matrix
(Ptable <- table(pred_P, Test_SVM_labels))
## Misclassification Rate for Polynomial
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))
#misclassification error is 48%


## Linear Kernel...
SVM_fit_L <- svm(Readmitted~., data=SVM_Train, kernel="linear", cost=.1, scale=FALSE)
print(SVM_fit_L)
##Prediction --
(pred_L <- predict(SVM_fit_L, SVM_Test_num, type="class"))
#confusion matrix
(L_table<-table(pred_L, Test_SVM_labels))
## Misclassification Rate for Linear
(MR_L <- 1 - sum(diag(L_table))/sum(L_table))
#misclassification error is 40%
qplot(Readmitted, pred_L, data=SVM_Test,  colour=Readmitted, geom = c("boxplot", "jitter"),
      main = "predicted vs. observed in validation data",
      xlab = "Observed Readmissions", ylab = "Predicted Readmissions")

## Radial Kernel...
SVM_fit_R <- svm(Readmitted~., data=SVM_Train, kernel="radial", cost=.1, scale=FALSE)
print(SVM_fit_R)
##Prediction --
(pred_R <- predict(SVM_fit_R, SVM_Test_num, type="class"))
#confusion matrix
(R_table<-table(pred_R, Test_SVM_labels))
## Misclassification Rate for Radial
(MR_R <- 1 - sum(diag(R_table))/sum(R_table))
# misclassification error is 43%

#best model with the least misclassification is the one with linear kernel. let's tune with different costs
tuned_cost <- tune(svm, Readmitted~., data=SVM_Train, kernel="linear", ranges=list(cost=c(.01,1,10,100)))
summary(tuned_cost)

##-------------------------TEXT MINING-------------------------------------------
library(tm)
library(wordcloud)
filename <- read.csv("Hospital_Reviews.csv")
MyData <- filename
str(filename)
#separate columns 1 and 2
Sent <- MyData$sentiment
print(Sent)
#remove it from the data
MyData <- MyData[-c(1)]
print(head(MyData))

library(tidyr)
MyData <- unite(MyData, "review")
head(MyData, n=5)
write.csv(MyData, "MyData.csv")

library(tm)
dfCorpus <- Corpus(VectorSource(MyData))
inspect(dfCorpus)
View(dfCorpus)

##Make everything lowercase
CleanCorpus <- tm_map(dfCorpus, content_transformer(tolower))
CleanCorpus <- tm_map(CleanCorpus, removePunctuation)
#Remove stopwords
CleanCorpus <- tm_map(CleanCorpus, removeWords, stopwords("english"))

#Document Stemming

## Stem document
CleanCorpus <- tm_map(CleanCorpus, stemDocument)
##Viewing the corpus content
CleanCorpus[[1]][1]


# Find the 20 most frequent terms: term_count
library(qdap)
term_count <- freq_terms(CleanCorpus,15)

# Plot 20 most frequent terms
plot(term_count)

#Term Document Matrix
review_tdm <- TermDocumentMatrix(CleanCorpus)
inspect(review_tdm)

# Convert TDM to matrix
review_m <- as.matrix(review_tdm)
# Sum rows and frequency data frame
review_term_freq <- rowSums(review_m)
# Sort term_frequency in descending order
review_term_freq <- sort(review_term_freq, decreasing = T)
# View the top 10 most common words
review_term_freq[1:10]

# Plot a barchart of the 20 most common words
barplot(review_term_freq[1:20], col = "steel blue", las = 2)

review_word_freq <- data.frame(term = names(review_term_freq), num = review_term_freq)
# Create a wordcloud for the values in word_freqs
wordcloud(review_word_freq$term, review_word_freq$num,
          max.words = 50, colors = "red")
# Print the word cloud with the specified colors
wordcloud(review_word_freq$term, review_word_freq$num,
          max.words = 50, colors = c("aquamarine","darkgoldenrod","tomato"))

# Tokenize descriptions
library(quanteda)

reviewtokens <- tokens(MyData$review,what="word", remove_numbers=TRUE,
                       remove_punct=TRUE, remove_symbols=TRUE, remove_hyphens=TRUE)
# Lowercase the tokens
reviewtokens=tokens_tolower(reviewtokens)
# remove stop words and unnecessary words
#rmwords <- c("and",)
reviewtokens=tokens_select(reviewtokens, stopwords(),selection = "remove")
#reviewtokens=tokens_remove(reviewtokens,rmwords)
# Stemming tokens
reviewtokens=tokens_wordstem(reviewtokens,language = "english")
reviewtokens=tokens_ngrams(reviewtokens,n=1:2)

# Creating a bag of words
reviewtokensdfm=dfm(reviewtokens,tolower = FALSE)
# Remove sparsity
reviewSparse <- convert(reviewtokensdfm, "tm")
tm::removeSparseTerms(reviewSparse, 0.7)
# Create the dfm
dfm_trim(reviewtokensdfm, min_docfreq = 0.3)
x=dfm_trim(reviewtokensdfm, sparsity = 0.98)

## Setup a dataframe with features
df=convert(x,to="data.frame")
## Add the Y variable Sentiment
reviewtokensdf_s <- cbind(Sent,df)
View(reviewtokensdf_s)

## Remove the original review.text column
reviewtokensdf_s=reviewtokensdf_s[,-c(2)]
View(reviewtokensdf_s)
reviewtokensdf_s$review <- as.factor(reviewtokensdf_s$review)

## Sentiment detection

#let's divide the dataset into training and test set by using sampling.

(n <- round(nrow(reviewtokensdf_s)/3))
(s <- sample(1:nrow(reviewtokensdf_s), n))
sent_Test <- reviewtokensdf_s[s,]
sent_Train <- reviewtokensdf_s[-s,]
#MAke sure the labels are out of the test data
sent_Test_noLabel<-sent_Test[-c(1)]
sent_Test_justLabel<-sent_Test$Sent
(head(sent_Test_noLabel))
#let's check how many rows are in the train and test dataset.
nrow(sent_Test)
nrow(sent_Train)
str(sent_Train)

#Run Naive Bayes model
## formula is label ~ x1 + x2 + .  NOTE that label ~. is "use all to create model"
library(e1071)
sent_NB_train<-naiveBayes(Sent~., data=sent_Train, na.action = na.pass)
sent_NB_pred <- predict(sent_NB_train, sent_Test_noLabel)
summary(sent_NB_train)
table(sent_NB_pred,sent_Test_justLabel)
confusionMatrix(sent_NB_pred, sent_Test_justLabel)
#60% accuracy

library(RWeka)
Gain <- InfoGainAttributeEval(Sent~., data = reviewtokensdf_s)
View(Gain)
# Sort Gain in descending order
Gain <- sort(Gain, decreasing = T)
barplot(Gain[1:10], col = "steel blue", las = 2)


#Run Naive Bayes model
## formula is label ~ x1 + x2 + .  NOTE that label ~. is "use all to create model"

sent_NB_train2<-naiveBayes(Sent~pcp+staff+facil, data=sent_Train, na.action = na.pass)
sent_NB_pred2 <- predict(sent_NB_train2, sent_Test_noLabel)
summary(sent_NB_train2)
sent_NB_table <- table(sent_NB_pred2,sent_Test_justLabel)
confusionMatrix(sent_NB_pred2, sent_Test_justLabel)
#accuracy improved to 70%


#let's see if SVM provides better results
## Radial Kernel...
SVM_fit_R <- svm(Sent~., data=sent_Train, kernel="radial", cost=0.01, scale=FALSE)
print(SVM_fit_R)
##Prediction --
(pred_R <- predict(SVM_fit_R, sent_Test_noLabel, type="class"))
#confusion matrix
(R_table<-table(pred_R, sent_Test_justLabel))
## Misclassification Rate for Radial
(MR_R <- 1 - sum(diag(R_table))/sum(R_table))
#accuracy is 40%

#####  We can "tune" this SVM model by altering the cost ####
SVM_fit_L <- svm(Sent~., data=sent_Train, kernel="linear", cost=0.01, scale=FALSE)
print(SVM_fit_L)
##Prediction --
(pred_L <- predict(SVM_fit_L, sent_Test_noLabel, type="class"))
#confusion matrix
(L_table<-table(pred_L, sent_Test_justLabel))
## Misclassification Rate for Radial
(MR_L <- 1 - sum(diag(L_table))/sum(L_table))
#accuracy is 50%

## Polynomial Kernel...
SVM_fit_P <- svm(Sent~., data=sent_Train, kernel="polynomial", cost=0.01, scale=FALSE)
print(SVM_fit_P)
##Prediction --
(pred_P <- predict(SVM_fit_P, sent_Test_noLabel, type="class"))
## COnfusion Matrix
(Ptable <- table(pred_P, sent_Test_justLabel))
## Misclassification Rate for Polynomial
(MR_P <- 1 - sum(diag(Ptable))/sum(Ptable))
#accuracy is 40%

# the best acccuracy rate was provided by linear kernel and at the best cost of 0.01.
tuned_cost <- tune(svm, Sent~., data=sent_Train, kernel="linear", ranges=list(cost=c(.01,1,10,100)))
summary(tuned_cost)

#build barplot to compare misclassification rates
MR <- data.frame("MR"=c(MR_R,MR_L,MR_P), "Models"=c("Radial","Linear","Polynomial") )
MR$Models <- as.character(MR$Models)
str(MR)
View(MR)
barplot(MR$MR,names.arg=c("Radial","Linear","Polynomial"))
