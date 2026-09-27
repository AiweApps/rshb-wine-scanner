"""Run the gateway: python -m wine_scanner (configuration from WINE_SCANNER_* environment variables)."""
import sys


def main():
    from wine_scanner.settings import Settings
    try:
        settings = Settings.from_env()
    except ValueError as error:
        sys.exit(f'wine_scanner: {error}')
    import uvicorn
    from wine_scanner.http import create_app
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False,
                proxy_headers=False, server_header=False)


if __name__ == '__main__':
    main()
