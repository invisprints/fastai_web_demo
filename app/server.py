import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

# export_file_url = 'https://drive.google.com/uc?export=download&id=1SjiG3ex0VWKQSqGjZJThVuFqK8Wuo_Lt'
export_file_url = 'https://www.kaggleusercontent.com/kf/22073878/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Uq5Wr5Ie4BLV3DEuWWmRHw.5Pdd8yvRde7cuddlTLMv4mAT7bcKORiwBSKWyAZ48BFw453A6CaXKrIsuL0eC2ZC3ev2iLHyIqw7l-65qmOzqbYVsZrLpNPRI9C-gOseGp4YXPP9-Et-J8YYfREKYTC37JKKkqugmCamo2syCy-RFyr2pGmiInJSkNDC8h46uzw.oIPh3wNDiDydfa8iUxXsLw/export.pkl'

export_file_name = 'export.pkl'
label_id_name_dict = {
'一次性快餐盒':'干垃圾',
'污损塑料':'干垃圾',
'烟蒂':'干垃圾',
'牙签':'干垃圾',
'破碎花盆及碟碗':'干垃圾',
'竹筷':'干垃圾',
'大骨头':'湿垃圾',
'剩饭剩菜':'湿垃圾',
'水果果皮':'湿垃圾',
'水果果肉':'湿垃圾',
'茶叶渣':'湿垃圾',
'菜叶菜根':'湿垃圾',
'蛋壳':'湿垃圾',
'鱼骨':'湿垃圾',
'包':'可回收物',
'充电宝':'可回收物',
'化妆品瓶':'可回收物',
'塑料玩具':'可回收物',
'塑料碗盆':'可回收物',
'塑料衣架':'可回收物',
'快递纸袋':'可回收物',
'插头电线':'可回收物',
'旧衣服':'可回收物',
'易拉罐':'可回收物',
'枕头':'可回收物',
'毛绒玩具':'可回收物',
'洗发水瓶':'可回收物',
'玻璃杯':'可回收物',
'皮鞋':'可回收物',
'砧板':'可回收物',
'纸板箱':'可回收物',
'调料瓶':'可回收物',
'酒瓶':'可回收物',
'金属食品罐':'可回收物',
'锅':'可回收物',
'食用油桶':'可回收物'.
'饮料瓶':'可回收物',
'干电池':'有害垃圾',
'软膏':'有害垃圾',
'过期药物':'有害垃圾'
}
# classes=[
#     "其他垃圾/一次性快餐盒",
#     "其他垃圾/污损塑料",
#     "其他垃圾/烟蒂",
#     "其他垃圾/牙签",
#     "其他垃圾/破碎花盆及碟碗",
#     "其他垃圾/竹筷",
#     "厨余垃圾/剩饭剩菜",
#     "厨余垃圾/大骨头",
#     "厨余垃圾/水果果皮",
#     "厨余垃圾/水果果肉",
#     "厨余垃圾/茶叶渣",
#     "厨余垃圾/菜叶菜根",
#     "厨余垃圾/蛋壳",
#     "厨余垃圾/鱼骨",
#     "可回收物/充电宝",
#     "可回收物/包",
#     "可回收物/化妆品瓶",
#     "可回收物/塑料玩具",
#     "可回收物/塑料碗盆",
#     "可回收物/塑料衣架",
#     "可回收物/快递纸袋",
#     "可回收物/插头电线",
#     "可回收物/旧衣服",
#     "可回收物/易拉罐",
#     "可回收物/枕头",
#     "可回收物/毛绒玩具",
#     "可回收物/洗发水瓶",
#     "可回收物/玻璃杯",
#     "可回收物/皮鞋",
#     "可回收物/砧板",
#     "可回收物/纸板箱",
#     "可回收物/调料瓶",
#     "可回收物/酒瓶",
#     "可回收物/金属食品罐",
#     "可回收物/锅",
#     "可回收物/食用油桶",
#     "可回收物/饮料瓶",
#     "有害垃圾/干电池",
#     "有害垃圾/软膏",
#     "有害垃圾/过期药物"]
# classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    output = learn.predict(img)[0]
#     prediction = classes[int(output.obj)]
    prediction = label_id_name_dict[ouput.obj]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
