import click
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    import app.models
    db.create_all()
    click.echo('Initialized the database.')


@click.command('drop-db')
@with_appcontext
def drop_db_command():
    """Clear all tables."""
    import app.models
    db.drop_all()
    click.echo('Drop all tables.')


def init_app(app):
    app.cli.add_command(init_db_command)
    app.cli.add_command(drop_db_command)
    db.init_app(app)