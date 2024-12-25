import factory from 'mxgraph';

(window as any)['mxBasePath'] = 'assets/mxgraph';

export default factory({
  // not working see https://github.com/jgraph/mxgraph/issues/479
  mxBasePath: 'assets/mxgraph',
});