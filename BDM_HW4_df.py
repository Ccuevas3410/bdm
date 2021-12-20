from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import datetime
import json
import numpy as np
import sys

def main(sc, spark):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]
    CAT_CODES = {'445210', '452311', '445120', '722410', '722511', '722513', '446110', '446191', '311811', '722515',
             '452210', '445220', '445230', '445291', '445292', '445299', '445110'}
    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446110': 5,
             '446191': 5, '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7,
             '445220': 7, '445292': 7, '445110': 8}

    dfD = dfPlaces.select("placekey","naics_code").filter(F.col("naics_code").isin(CAT_CODES))

    udfToGroup = F.udf(lambda x : CAT_GROUP[x])
    dfE = dfD.withColumn('group', udfToGroup('naics_code'))

    dfF = dfE.drop('naics_code').cache()

    groupCount = dict(dfF.groupby("group").count().sort("group").collect())

    def expandVisits(date_range_start, visits_by_day):
        d = datetime.datetime.strptime(date_range_start,f"%Y-%m-%dT%H:%M:%S%z").date()
        visits = visits_by_day.replace("[","").replace("]","").split(",")
        result = []
        for v in visits:
            result.append((d.year,d.strftime(f"%m-%d"),int(v)))
            d+=datetime.timedelta(days=1)
        return tuple(result)
    
    visitType = T.StructType([T.StructField('year', T.IntegerType()),T.StructField('date', T.StringType()),T.StructField('visits', T.IntegerType())])
    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))
    dfH = dfPattern.join(dfF, 'placekey').withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))).select('group', 'expanded.*')

    def computeStats(group, visits):
        median = max(int(np.median(visits)),0)
        low = max(int(np.median(visits)-np.std(visits)),0)
        high = max(int(np.median(visits)+np.std(visits)),0)
        return (median,low,high)
    
    statsType = T.StructType([T.StructField('median', T.IntegerType()),T.StructField('low', T.IntegerType()),T.StructField('high', T.IntegerType())])
    udfComputeStats = F.udf(computeStats, statsType)
    dfI = dfH.groupBy('group', 'year', 'date').agg(F.collect_list('visits').alias('visits')).withColumn('stats', udfComputeStats('group', 'visits'))
    udfAddPrefix = F.udf(lambda x: "2020-"+x)
    dfJ = dfI.select("group","year",udfAddPrefix("date").alias("date"),"stats.*").sort("group","year","date").cache()

    
    filenames = {0:"big_box_grocers",1:"convenience_stores",2:"drinking_places",3:"full_service_restaurants",
             4:"limited_service_restaurants",5:"pharmacies_and_drug_stores",6:"snack_and_bakeries",
             7:"specialty_food_stores",8:"supermarkets_except_convenience_stores"}
    for i in filenames.keys():
        dfJ.filter(f'group={i}') \
            .drop('group') \
            .coalesce(1) \
            .write.csv(f'/content/{filenames[i]}',mode='overwrite', header=True)
    
if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)
