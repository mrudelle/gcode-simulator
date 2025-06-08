#!/usr/bin/env python3
import click
import datetime
import json
import sys

from .gcode_simulator import GCodeSimulator, GrblSettings
from .viz import plot_trace


@click.command(context_settings={'show_default': True})
@click.argument(
    'gcode_file', type=click.Path(exists=True, file_okay=True, readable=True)
)
@click.option(
    '--max-rate-x',
    type=float,
    default=3000.0,
    help='Maximum feed rate for X axis in mm/min ($110)',
)
@click.option(
    '--max-rate-y',
    type=float,
    default=3000.0,
    help='Maximum feed rate for Y axis in mm/min ($111)',
)
@click.option(
    '--max-accel-x',
    type=float,
    default=800.0,
    help='Maximum acceleration for X axis in mm/s^2 ($120)',
)
@click.option(
    '--max-accel-y',
    type=float,
    default=800.0,
    help='Maximum acceleration for Y axis in mm/s^2 ($121)',
)
@click.option(
    '--junction-deviation',
    type=float,
    default=0.01,
    help='Junction deviation in mm ($11)',
)
@click.option('--grbl-version', type=str, default='0.9i', help='GRBL firmware version')
@click.option(
    '--visualize/--no-visualize',
    default=False,
    help='Display a visualization of the toolpath',
)
@click.option(
    '--json-output/--no-json-output',
    default=False,
    help='Output results in JSON format',
)
def main(
    gcode_file,
    max_rate_x,
    max_rate_y,
    max_accel_x,
    max_accel_y,
    junction_deviation,
    grbl_version,
    visualize,
    json_output,
):
    settings = GrblSettings(
        max_rate_x=max_rate_x,
        max_rate_y=max_rate_y,
        max_accel_x=max_accel_x,
        max_accel_y=max_accel_y,
        junction_deviation=junction_deviation,
        grbl_version=grbl_version,
    )

    simulator = GCodeSimulator(settings, trace=visualize)

    try:
        with open(gcode_file, 'r') as f:
            gcode = f.read()
    except Exception as e:
        click.echo(f'Error reading file: {e}', err=True)
        sys.exit(1)

    time_seconds, bounds = simulator.estimate_time(gcode)

    time_formatted = str(datetime.timedelta(seconds=round(time_seconds)))

    if json_output:
        result = {
            'execution_time': {'seconds': time_seconds, 'formatted': time_formatted},
            'bounds': {
                'x': {'min': bounds.min_x, 'max': bounds.max_x, 'width': bounds.width},
                'y': {
                    'min': bounds.min_y,
                    'max': bounds.max_y,
                    'height': bounds.height,
                },
            },
            'grbl_settings': {
                'max_rate_x': max_rate_x,
                'max_rate_y': max_rate_y,
                'max_accel_x': max_accel_x,
                'max_accel_y': max_accel_y,
                'junction_deviation': junction_deviation,
                'grbl_version': grbl_version,
            },
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f'Estimated execution time: {time_formatted}')
        click.echo('Bounds:')
        click.echo(
            f'  X: {bounds.min_x:.2f} to {bounds.max_x:.2f} (width: {bounds.width:.2f}mm)'
        )
        click.echo(
            f'  Y: {bounds.min_y:.2f} to {bounds.max_y:.2f} (height: {bounds.height:.2f}mm)'
        )

    if visualize:
        plot_trace(simulator.trace_nodes, bounds)


if __name__ == '__main__':
    main()
