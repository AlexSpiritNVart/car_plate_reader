import torch


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes

def box_select(boxes, xmin, ymin, xmax, ymax):
    '''Select boxes in range (xmin,ymin,xmax,ymax).

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    '''
    mask = (boxes[:,0]>=xmin) & (boxes[:,1]>=ymin) \
         & (boxes[:,2]<=xmax) & (boxes[:,3]<=ymax)
    boxes = boxes[mask,:]
    return boxes, mask

def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def box_nms(bboxes, scores, threshold=0.5, min_area=10):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1) #area every box
    
#     print('SCORES into NMS', scores.data.cpu().numpy())
#     print('AREAS into NMS', areas.data.cpu().numpy())
    
    # want to keep smallest ones or most sure ones
#         _, order = areas.sort(0, descending=False)
        
        
    _, order = scores.sort(0, descending=True)
    
#     area_mask = areas>240
#     areas = areas[area_mask]
#     print('AREAS into NMS', areas.data.cpu().numpy())

#     x1=x1[area_mask]
#     x2=x2[area_mask]
#     y1=y1[area_mask]
#     y2=y2[area_mask]
    

    
#     #disables nms
#     return torch.tensor(order, dtype=torch.long)
#     print('ORDER into NMS', order.data.cpu().numpy())
    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            if areas[order]>min_area:
                keep.append(order)
#             print('Cant read order[0]. current order ',order.data.cpu().numpy())
            break
        keep.append(i)
#        if areas[i]>min_area:
#             print('Nms appending')
#            keep.append(i)
#        else:
#            order = order[1:]
#            continue
        
#         print('NOT STABLElen(keep)!@#!@!!!!!!!!!!!!! ',len(keep))

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item()) #x1 s euqal or bigger than x1[i]
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())
        # like watching how can we shrink the i bbox

        w = (xx2-xx1).clamp(min=0) #width for every shrink variant
        h = (yy2-yy1).clamp(min=0) # height or every shrink variant
        inter = w * h # area every shrink var

        overlap = inter / (areas[i] + areas[order[1:]] - inter) # shrinkArea/(detectedArea[i] + detectedArea(shrinking)- shrinkArea)
        # its like Intersection over union
        
        ids = (overlap<=threshold).nonzero().squeeze()
        # we get ids overlapping with i bbox less?? thon on 0.5 
        # it will be like mask [0,0,1,0,1] but without i itself
        #than nonzero will make it [2,4]
        # so we just killed all bboxes except i and those overlapping not much 
        
#         print('Ids into NMS', ids.data.cpu().numpy())
        if ids.numel() == 0:
#             print('no interesting(unique) bboxes left')
            break
        order = order[ids+1]
#         print('new order ',order.data.cpu().numpy(),' with numel ',order.numel())
#         if order.numel() == 1:
            
        #than we keep only interesting bboxes
        #+1 because we didnt count one that was i th
     
    return torch.tensor(keep, dtype=torch.long)
