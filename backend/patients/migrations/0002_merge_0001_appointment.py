# Generated manually for Django migration merge

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('patients', '0001_appointment'),
        ('patients', '0002_appointment'),
    ]

    operations = [
        # This is a merge migration to resolve conflicts
        # Both 0001_appointment and 0002_appointment create the same Appointment model
        # The actual table creation should already be done, so this just merges the history
    ]
