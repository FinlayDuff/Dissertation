from .dataset_prep import (
    transform_fakes_dataset,
    transform_politifact_dataset,
    tranform_recovery_news_dataset,
    transform_isot_dataset,
    transform_covid_fake_news,
)

TRANSFORM_FUNCTIONS = {
    "FA-KES": transform_fakes_dataset,
    "recovery-news-data": tranform_recovery_news_dataset,
    "politifact": transform_politifact_dataset,
    "isot": transform_isot_dataset,
    'covid_fake_news': transform_covid_fake_news,
}
