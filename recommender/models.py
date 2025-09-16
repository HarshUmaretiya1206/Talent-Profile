from django.db import models


class Talent(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255, blank=True, default='')
    gender = models.CharField(max_length=64, blank=True, default='')
    monthly_rate = models.CharField(max_length=64, blank=True, default='')
    hourly_rate = models.CharField(max_length=64, blank=True, default='')

    class Meta:
        verbose_name = 'Talent'
        verbose_name_plural = 'Talents'

    def __str__(self) -> str:
        return self.name


