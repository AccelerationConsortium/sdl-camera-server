import argparse
import json
import multiprocessing
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser(description='USB and RealSense camera service')
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',default=8070,type=int)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    config['_lock_path']=str(args.config.resolve())+'.lockfile'
    from .api import create_app
    import uvicorn
    uvicorn.run(create_app(config),host=args.host,port=args.port,workers=1)


if __name__=='__main__':
    main()
