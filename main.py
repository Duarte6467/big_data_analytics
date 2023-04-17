from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import os
import pandas as pd

os.environ['SPARK_LOCAL_IP'] = "192.168.0.10"  # Set local ipv4 Address

sc = SparkContext.getOrCreate()
print(sc.version)

# create a SparkSession
spark = SparkSession.builder.appName("csv_file").getOrCreate()

# Define the schema of the DataFrame
schema = StructType([
    StructField("Country Name", StringType(), True),
    StructField("Regional Indicator", StringType(), True),
    StructField("Year", IntegerType(), True),
    StructField("Life Ladder", FloatType(), True),
    StructField("Log GDP Per Capita", FloatType(), True),
    StructField("Social Support", FloatType(), True),
    StructField("Healthy Life Expectancy At Birth", FloatType(), True),
    StructField("Freedom To Make Life Choices", FloatType(), True),
    StructField("Generosity", FloatType(), True),
    StructField("Perceptions Of Corruption", FloatType(), True),
    StructField("Positive Affect", FloatType(), True),
    StructField("Negative Affect", FloatType(), True),
    StructField("Confidence In National Government", FloatType(), True)
])

# Read the CSV file with the specified schema
df = spark.read.format("csv") \
    .option("header", True) \
    .option("nullValue", "") \
    .schema(schema) \
    .load("happiness.csv")

# show the DataFrame
df.printSchema()
df.show()

# Columns to remove for Data Analysis

column_summary = df.drop("Country Name", "Regional Indicator", "Year")

print("Number of observations", df.count())

# Count the number of null values in each column
missing_count = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])  # (NNK, 2021)

# Show how many missing values in each column
missing_count.show()

# https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.summary.html
column_summary = df.drop("Country Name", "Regional Indicator", "Year")
column_summary.summary().show()

"""
# Filter based on Condition
filtered_df = df.filter(col('Country Name') == 'Guinea')

# show the filtered DataFrame
filtered_df.show()
"""

# Drop Missing Values on Regional Indicator
df = df.dropna(subset=["Regional Indicator"])
print("New number of observations:", df.count())

missing_count = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])  # (NNK, 2021)

# Show how many missing values in each column
missing_count.show()

# (Apache, n.d.)
column_summary = df.drop("Country Name", "Regional Indicator", "Year")  # No use in asssesing these columns
column_summary.describe().show()

# Perform the Process of filling the missing values


# assume you have a DataFrame called 'df' with columns 'Country Name', 'Year', and other columns

# Define a Window, that is divided by 'Country Name' (Davis, 2022)
w = Window.partitionBy('Country Name')

# Fill missing values with the mean of each column for each country
df = df.select("Country Name", "Regional Indicator",
               *[when(col(c).isNull(), avg(col(c)).over(w)).otherwise(col(c)).alias(c) for c in df.columns[2:]])

# Show how many missing values in each column
missing_count = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])  # (NNK, 2021)
missing_count.show()

# Show 100 rows
df.show(100)

columns = ["Life Ladder", "Log GDP Per Capita", "Social Support", "Healthy Life Expectancy At Birth",
           "Freedom To Make Life Choices", "Generosity", "Perceptions Of Corruption", "Positive Affect",
           "Negative Affect", "Confidence In National Government"]

# (How to Get the Correlation Matrix of a Pyspark Data Frame?, 2018)
# Assemble the columns into a vector column
vectorAssembler = VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="skip")
vector_df = vectorAssembler.transform(df).select("features")

# Calculate the correlation matrix
correlation_matrix = Correlation.corr(vector_df, "features").head()

# Convert to Array
corr_matrix = correlation_matrix[0].toArray()

corr_df = pd.DataFrame(corr_matrix, columns=columns, index=columns)  # Converted to Pandas Dataframe
# Print the correlation matrix
pd.set_option("display.max_columns", None)
print(corr_df)

""" Linear Regression Section. The following code was based on (Hejazi, 2023), and derived using some other parameters in Pyspark Documnetation"""

# Define the list of determinants and the dependent variable
determinants = ["Log GDP Per Capita", "Freedom To Make Life Choices", "Generosity", "Perceptions Of Corruption",
                "Negative Affect", "Positive Affect", "Social Support"]
target = "Life Ladder"

# Combine the determinants into a single features column
assembler = VectorAssembler(inputCols=determinants, outputCol="features", handleInvalid="skip")

# Transform the df
data = assembler.transform(df)

# Split data into training, validation and testing
(trainingData, validationData, testData) = data.randomSplit([0.6, 0.2, 0.2],
                                                            seed=42)  # (pyspark.sql.DataFrame.randomSplit — PySpark 3.1.3 Documentation, n.d.)

# Define initial values and set of possibilities for the loop to iterate
reg_params = np.arange(1, 50, 1)
best_mse = float("inf")
best_reg_param = None

for reg_param in reg_params:  # Loop through the regParam values
    # Train a linear regression model with the current regParam value
    lr = LinearRegression(featuresCol="features", labelCol=target, solver="auto", fitIntercept=True,
                          regParam=reg_param)  # (LinearRegression, n.d.)
    model = lr.fit(trainingData)

    # Make predictions on the validation set
    predictions = model.transform(validationData)  # (ML Tuning - Spark 3.4.0 Documentation, n.d.)

    # Evaluate the Mean Squared Error (MSE) on the validation set
    evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)

    if mse < best_mse:
        best_mse = mse  # If current Mean Squared Error is lower, it becomes the Best MSE
        best_reg_param = reg_param  # If the current MSE is lower, then current RegParam is the best

# Combine the training and validation sets
train_val_data = trainingData.union(validationData)  # Increase number of samples to improve performance

# Fit the final linear regression model using the best regParam on the combined Training/Validation datasets
lr = LinearRegression(featuresCol="features", labelCol=target, solver="auto", fitIntercept=True,
                      regParam=best_reg_param)
model = lr.fit(train_val_data)

# Fetch Summary Values
summary = model.summary
# Fetch P-values
p_values = summary.pValues

# Make predictions on the testing data. This is done to evaluate the model's performance on unseen data
# and ensure that the MSE of the combined training/validation set is similar to the testing set.
predictions = model.transform(testData)

# Evaluate the model's mean squared error (MSE) on the testing set
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="mse")
test_mse = evaluator.evaluate(predictions)

# Print the results
print("Best regParam:", best_reg_param)
print("Coefficients:", list(zip(determinants, model.coefficients)))
print("P-Values:", list(zip(determinants, p_values)))
print("R Squared:", summary.r2)
print("Training and Validation Mean Squared Error:", best_mse)
print("Testing Mean Squared Error:", test_mse)
#################################################################


# This method did not work
# df.coalesce(1).write.csv("cleaned_dataset.csv", mode="overwrite", header=True)

# Convert the Dataset to pandas so that the csv is saved as only one File.
csv = df.toPandas()
csv.to_csv("cleaned_dataset.csv", index=False)
spark.stop()
