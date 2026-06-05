import os
import sys

from pyspark.sql import SparkSession


_spark = None


def get_spark(app_name="COMP5434"):
    global _spark

    if _spark is not None:
        return _spark

    active = SparkSession.getActiveSession()
    if active is not None:
        _spark = active
        return _spark

    # Ensure PySpark workers use the same Python as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    _spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.broadcastTimeout", "600")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    return _spark
