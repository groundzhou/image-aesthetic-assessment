import click
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    db.create_all()
    click.echo('Initialized the database.')


@click.command('drop-db')
@with_appcontext
def drop_db_command():
    """Clear all tables."""
    db.drop_all()
    click.echo('drop all tables.')


def init_app(app):
    app.cli.add_command(init_db_command)
    db.init_app(app)
