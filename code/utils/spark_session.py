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

    _spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.broadcastTimeout", "600")
        .getOrCreate()
    )
    return _spark
