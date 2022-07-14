import boto3
import sagemaker


class S3Connector:
    """
    Class pulls data from source s3 bucket that might be a public bucket. Saves file to 
    local disk. The user can than upload the file to the users s3 bucket
    """
    def __init__(self, sagemaker_session, project_name):
        # Do we need sagemaker session here? Use s3 resource to upload file?
        self.sagemaker_session = sagemaker_session
        self.project_name = project_name
    
    def get_from_s3(self, bucket, file, local_path):
        """Downlod data from s3 bucket to local disk
        """
        s3 = boto3.resource("s3")
        region = boto3.Session().region_name
        s3.Bucket(f"{bucket}-{region}").download_file(
            f"{file}", local_path
        )
        return self

    def upload_to_s3(self, local_path, bucket_name=None):
        """Upload data to s3 bucket
        """
        if bucket_name is None:
            # upload to default sagemaker bucket
            bucket_name = self.sagemaker_session.default_bucket()
            base_uri = f"s3://{bucket_name}/{self.project_name}"
        
        sagemaker.s3.S3Uploader.upload(
            local_path=local_path,
            desired_s3_uri=base_uri,
        )


if __name__ == '__main__':
    sagemaker_session = sagemaker.session.Session()
    source_parameters = {
        "bucket": "sagemaker-servicecatalog-seedcode",
        "file": "dataset/abalone-dataset.csv",
        "local_path": "abalone-dataset.csv"
    }
    connector = S3Connector(sagemaker_session, project_name='abalone')
    connector\
        .get_from_s3(**source_parameters)\
        .upload_to_s3(local_path="abalone-dataset.csv")
