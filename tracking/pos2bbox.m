function bbox_out = pos2bbox(pos, sz)

tl = pos - (sz - 1)/2;
br = pos + (sz - 1)/2;
x1 = tl(2); y1 = tl(1);
x2 = br(2); y2 = br(1);

bbox_out = round([y1, y2, x1, x2]);

end