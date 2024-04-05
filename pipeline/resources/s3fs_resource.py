import s3fs

from dagster import ConfigurableResource, InitResourceContext


class S3FSResource(ConfigurableResource):
    key: str
    secret: str
    endpoint_url: str
    use_ssl: str
    region_name: str

    def create_resource(self, context: InitResourceContext):
        fs = s3fs.S3FileSystem(
            key=self.key,
            secret=self.secret,
            endpoint_url=self.endpoint_url,
            use_ssl=self.use_ssl,
            client_kwargs={
                "region_name": self.region_name,
            },
        )
        return fs
