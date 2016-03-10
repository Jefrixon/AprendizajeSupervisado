###PAQUETES QUE SE USARON:###
#- install.packages('rpart.plot')
#- install.packages('caret')
#- install.packages('e1071')
#- install.packages('RWeka')
#- install.packages('som')
#- install.packages('class')
#- install.packages('pROC')

# Librerias
library ('rpart')
library ('rpart.plot')
library('caret')
library('e1071')
library('RWeka')
library('som')
library('class')
library('pROC')

# Lectura de los datos ----
getwd()
setwd("~/CosasDeLaUniversidad/Semestre VIII/Mineria/Tarea2")
raw_data <- read.csv2(
  file = "minable.csv",
  header = T,sep = ",")

# Quitamos las variables inecesarias!

raw_data$cIdentidad <- NULL
raw_data$fNacimiento <- NULL
raw_data$eCivil <- NULL
raw_data$jReprobadas <- NULL
raw_data$pReside <- NULL
raw_data$dHabitacion <- NULL
raw_data$cDireccion <- NULL
raw_data$oSolicitudes <- NULL
raw_data$aEconomica<- NULL
raw_data$sugerencias <- NULL
raw_data$rating <- NULL

# Quitar las variables "levels" que no dejan que se haga el rpart
raw_data$pAprobado <- as.numeric(as.character(raw_data$pAprobado))
raw_data$eficiencia <- as.numeric(as.character(raw_data$eficiencia))
raw_data$irMensual <- as.numeric(as.character(raw_data$irMensual))
raw_data$irOtros <- as.numeric(as.character(raw_data$irOtros))
raw_data$grMedicos <- as.numeric(as.character(raw_data$grOdontologicos))
raw_data$grCondominio <- as.numeric(as.character(raw_data$grCondominio))
raw_data$grOtros <- as.numeric(as.character(raw_data$grOtros))
raw_data$grOdontologicos <- as.numeric(as.character(raw_data$grOdontologicos))

# Cambiar los NA de la fila 97 por la media de la columna respectiva
raw_data$grMedicos[97] = summary(raw_data$grMedicos)[4]
raw_data$grOdontologicos[97] = summary(raw_data$grOdontologicos)[4]

# Se procede a hacer un proceso de estratificación
datos_mIngreso_cero <- raw_data[raw_data$mIngreso==0,]
datos_mIngreso_uno <- raw_data[raw_data$mIngreso==1,]
datos_mIngreso_dos <- raw_data[raw_data$mIngreso==2,]
datos_mIngreso_tres <- raw_data[raw_data$mIngreso==3,]

# no se toma en cuenta el registro el cual mIngreso es 1, ya que la matriz de
# confución no queda cuadrada.
set.seed(14)
samp = sample(2,nrow(datos_mIngreso_cero), replace = TRUE, prob = c(0.8,0.2))
entrenamiento_cero = datos_mIngreso_cero[samp==1,]
prueba_cero = datos_mIngreso_cero[samp==2,]

samp = sample(2,nrow(datos_mIngreso_dos), replace = TRUE, prob = c(0.8,0.2))
entrenamiento_dos = datos_mIngreso_dos[samp==1,]
prueba_dos = datos_mIngreso_dos[samp==2,]

samp = sample(2,nrow(datos_mIngreso_tres), replace = TRUE, prob = c(0.8,0.2))
entrenamiento_tres = datos_mIngreso_tres[samp==1,]
prueba_tres = datos_mIngreso_tres[samp==2,]

entrenamiento = rbind(entrenamiento_cero,entrenamiento_dos,entrenamiento_tres)
prueba = rbind(prueba_cero,prueba_dos,prueba_tres)


## Siguiente paso, crear el arbol de desición y buscamos el mejor arbol posible.
minsplits <- c(2,5,10,50,300,1000)
cps <- c(0.3,0.2,0.1,0.0001,0.00000001)
minbuckets <- c(2,5,10,50,300)
arbol_exactitudes <- list()
for (i in minsplits){
  for(j in cps){
    for(k in minbuckets){
      arbol<-rpart(mIngreso ~ . , data = entrenamiento, method = 'class',
                  control = rpart.control(minsplit = i, minbucket = k, cp = j))
      prediccion <- predict(arbol, newdata = prueba, type = "class")
      theConfusionMatrix <- table(prediccion, prueba$mIngreso)
      arbol_confusionMatrix <- confusionMatrix(prediccion, prueba$mIngreso)
      exactitud <- arbol_confusionMatrix$overall[1]
      arbol_exactitudes <- rbind(c(arbol_exactitudes,i,j,k,exactitud))
      
    }
  }
}
arbol_exactitudes <- matrix(arbol_exactitudes,ncol = 4,byrow = T)
header <-c("minsplits","cp","minbuckets","accuracy")
colnames(arbol_exactitudes) <- header
arbol_exactitudes <- as.data.frame(arbol_exactitudes)
max_arbol_exactitudes <- arbol_exactitudes[arbol_exactitudes$accuracy>= max(unlist(arbol_exactitudes$accuracy)),]

# El Siguiente arbol, es el "optimo"
arbol<-rpart(mIngreso ~ . , data = entrenamiento, method = 'class',
             control = rpart.control(minsplit = max_arbol_exactitudes[1,]$minsplits,
                                     minbucket = max_arbol_exactitudes[1,]$minbuckets,
                                     cp = max_arbol_exactitudes[1,]$cp))
prediccion_arbol <- predict(arbol, newdata = prueba, type = "class")
theConfusionMatrix <- table(prediccion_arbol, prueba$mIngreso)
arbol_confusionMatrix <- confusionMatrix(prediccion_arbol, prueba$mIngreso)
exactitud_arbol <- arbol_confusionMatrix$overall[1]
#rpart.plot(arbol)

## Reglas de clasificación
entrenamiento_clasificacion <- entrenamiento
prueba_clasificacion <- prueba
entrenamiento_clasificacion$mIngreso <- as.factor(entrenamiento_clasificacion$mIngreso)
prueba_clasificacion$mIngreso <- as.factor(prueba_clasificacion$mIngreso)

rClasificacion = JRip(formula = mIngreso ~ ., data = entrenamiento_clasificacion)
prediccion_clasificacion <- predict(rClasificacion, newdata = prueba_clasificacion,type = "class")
rClasificacion_confucionMatrix = table(prueba_clasificacion$mIngreso, prediccion_clasificacion)
rClasificacion_confucionMatrix <- confusionMatrix(rClasificacion_confucionMatrix)
exactitud_rCLasificacion <- rClasificacion_confucionMatrix$overall[1]

## KNN
#normalizar los valores
#raw_data$grMedicos[97] = summary(raw_data$grMedicos)[4]
#raw_data$grOdontologicos[97] = summary(raw_data$grOdontologicos)[4]

knn_entrenamiento <- entrenamiento
knn_prueba <- prueba
knn_entrenamiento_n <- normalize(knn_entrenamiento, byrow=TRUE)
knn_prueba_n <- normalize(knn_prueba, byrow=TRUE)

#predicción Knn
knn_prediccion <- knn(train = knn_entrenamiento_n, test = knn_prueba_n, cl = knn_entrenamiento$mIngreso, k=14)
knn_confucionMatrix = table(knn_prueba$mIngreso, knn_prediccion)
knn_confucionMatrix <- confusionMatrix(knn_confucionMatrix)
exactitud_knn <- knn_confucionMatrix$overall[1]

niveles_prueba <- knn_prueba$mIngreso
# análisis ROC
# ROC Arbol
roc_arbol <- roc(prueba$mIngreso, as.numeric(prediccion_arbol), levels = niveles_prueba)
plot(roc_arbol)

# ROC Reglas de Clasificación
roc_rClasificacion <- roc(prueba_clasificacion$mIngreso, as.numeric(prediccion_clasificacion), levels = niveles_prueba)
plot(roc_rClasificacion)

# ROC Knn
roc_knn <- roc(knn_prueba$mIngreso, as.numeric(knn_prediccion), levels = niveles_prueba)
plot(roc_knn)
